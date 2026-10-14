/**
 * 对话生成（host 半侧）：用 DSH 的 LLM 统一抽象层（ctx.llm）按条目配置的 chatModel
 * （留空 = 当前对话用的 provider/model）生成一句回复。与碎碎念（generateWhisper）同构，区别是：
 *  - 输入带历史对话（memory.json 截取的最近 N 轮），历史以 user/assistant 消息进入请求；
 *  - user 消息 = 用户刚输入的话（不是"随便叨叨"指令）；
 *  - 回复放宽到 256 token（对话比碎碎念可说得稍多），超时放宽到 60s。
 *
 * 设计：
 *  - provider/model：条目配置的 chatModel 优先（留空 = 不指定），失败回落到
 *    agentDefaultModel.currentSelection()（当前对话的模型，与余额/碎碎念同源）重试一次——
 *    候选链见 model-selection.ts；
 *  - system = 用户配置的 whisperPrompt（人设：碎碎念与对话共用同一人设）；
 *  - reasoningEffort: 'off' —— 仅当模型声明支持 reasoning effort（含 "off"）时传，
 *    关闭深度思考：闲聊对话不需要推理。无 reasoning 元数据的模型（如
 *    reasoningEfforts: false）显式传 off 会被 dsh-llm 判为 UNSUPPORTED_REASONING_EFFORT
 *    并折叠成空流（表现为"模型未返回文本"），因此这类模型省略该字段（语义等价于不传）；
 *  - sessionId: PET_CALL_IDENTITY —— 调用身份（见 llm-call.ts）；
 *  - 无正文时读 BlockAssembler.finish 报真实失败原因（dsh-llm 把适配器抛错折进终止块）；
 *  - 历史 assistant 消息用 createAssistantMessage 构造（provider/model 记当前选择，
 *    仅作消息角色载体，不涉及适配器回放）；
 *  - 流式收集 + BlockAssembler 拼装文本；生成失败显式返回结构化原因，不伪造文案。
 *
 * 配图（可选，pool 非空时）：与碎碎念「随机抽」不同——对话有真实上下文（用户输入 + 历史），
 * 故把整张表情包清单交给模型由它**按语境选**一张；模型在末尾附标记 [图:名称]，
 * host 解析并**只在池内命中时**采纳（防幻觉出池外名称），未选/选错则不配图
 * （对话配图是点缀，不强制每句都带）。
 */

import {
  BlockAssembler,
  createAssistantMessage,
  createUserMessage,
  ReasoningEffortId,
  type ContentBlock,
  type ImageBlock,
} from '@deepseek-ai/dsh-llm';
import { PET_CALL_IDENTITY, noTextReason } from './llm-call';
import { supportsReasoningOff } from './llm-reasoning';
import { extractChatImage, memeCatalog, type MemeEntry } from './memes';
import { modelCandidates, modelLabel, type ModelRef } from './model-selection';

/** 生成失败原因（与 shared/whisper.ts 的 WhisperState 失败分支同构） */
export type ChatGenerateResult =
  | { ok: true; text: string; image?: string }
  | { ok: false; reason: 'provider-missing' | 'generate-error'; message?: string };

/** 记忆中的一条消息（与 shared/chat.ts 的 ChatMessage 同构） */
export interface ChatMemoryMessage {
  role: 'user' | 'assistant';
  content: string;
  ts: number;
}

/**
 * 随本轮提问附带的截图（已存为 attachment，走 dsh-llm 的 image 块）。
 *
 * 【为什么必须是 attachment 引用而不是 base64】dsh-llm 的 ImageBlock 只认
 * `ImageAttachmentRef`——图片要先进 dsh-attachment 的 store 拿到内容寻址引用，
 * 由请求组装期按路由预算再编码。直接把 base64 塞进 content 不是合法块，会被丢。
 */
export interface ChatImageInput {
  /** 截图 JPEG 的 base64（无 data: 前缀）；由 host 存入 attachment 换成 ref */
  data: string;
  /** 媒体类型；截屏固定 image/jpeg */
  mediaType: string;
}

/**
 * 本插件用到的附件服务子集（`ctx.attachments`）。
 *
 * 【为什么用结构化类型而不是 import AttachmentStore】`@deepseek-ai/dsh-attachment`
 * 只发 JS、不带 .d.ts，import 它必然 TS2307；而为一个方法新增一条 peerDependency 会
 * 让插件多一个装不上就装不上的硬依赖。这里只声明真正调用的那一个方法，靠结构化类型
 * 与实现方兼容——ctx.attachments 是注入的服务实例，鸭子类型匹配。
 */
interface AttachmentStoreLike {
  saveImages(
    inputs: ReadonlyArray<{ data: Uint8Array; mediaType: string; name?: string }>,
  ): Promise<readonly unknown[]>;
}

/** 单次生成超时（ms）：对话等 LLM 回复，60s 足够 */
const TIMEOUT_MS = 60_000;

/** 配图指令：附在 user 正文之后（紧邻回答位置，模型更容易遵守） */
function imageInstruction(pool: MemeEntry[]): string {
  return (
    '\n\n[配图] 回复结尾可选附一张表情包给用户看，从下列清单里挑最贴合当前语境的：\n' +
    memeCatalog(pool) +
    '\n挑中就在回复最后另起一行写 [图:名称]（名称原样照抄）；没有合适的就完全不要写这个标记。'
  );
}

/**
 * 生成一句对话回复。
 * @param ctx 宿主上下文（注入 agentDefaultModel / llm）
 * @param system 人设提示词（whisperPrompt）
 * @param history 最近记忆（按时间正序；user/assistant 交替）
 * @param userText 用户刚输入的话
 * @param pool 表情包候选池（开启对话配图时传入；空/缺省 = 纯文本，指令与解析都不介入）
 * @param preferred 条目配置单独指定的模型（chatModel）；缺省 / 留空 = 只用当前对话的模型
 * @param image 随本轮提问附带的截图；缺省 = 纯文本提问（与旧行为逐字一致）
 * @param memorySummary 「很久之前」的记忆摘要（记忆整理产出）；缺省 = 无摘要、
 *   上下文里不出现摘要段（与旧行为逐字一致）。**不是**历史消息，故不放进 history 数组——
 *   它进的是 system 尾部，语义是「背景设定」而不是「刚说过的话」，模型不会把它当新一轮对话回应。
 * @returns 回复文本（+ 命中池内的配图名），或结构化失败（provider 缺失 / 生成错误）
 */
export async function generateChat(
  ctx: {
    agentDefaultModel: { currentSelection(): { provider: string; model: string } };
    llm?: unknown;
    attachments?: AttachmentStoreLike;
  },
  system: string,
  history: ChatMemoryMessage[],
  userText: string,
  pool: MemeEntry[] = [],
  preferred?: ModelRef,
  image?: ChatImageInput,
  memorySummary?: string,
): Promise<ChatGenerateResult> {
  // 候选链：配置的模型优先 → 当前对话的模型兜底（chatModel 留空时链上只有后者，与旧版逐字一致）
  const candidates = modelCandidates(ctx, preferred);
  if (candidates.length === 0) {
    return { ok: false, reason: 'provider-missing', message: '当前对话未配置模型' };
  }
  const llm = (ctx as { llm?: { stream(o: unknown): AsyncIterable<unknown> } }).llm;
  if (!llm || typeof llm.stream !== 'function') {
    return { ok: false, reason: 'generate-error', message: 'LLM 服务不可用' };
  }
  // 截图先存成 attachment 拿引用：**只存一次**，不随候选链重试重复写盘（内容寻址，重试也只多一次
  // 哈希比对）。存失败要显式报错而不是静默降级成纯文本提问——用户明确截了图，悄悄丢掉比报错更糟。
  let imageRef: unknown;
  if (image) {
    const store = ctx.attachments;
    if (!store || typeof store.saveImages !== 'function') {
      return { ok: false, reason: 'generate-error', message: '附件服务不可用，无法发送截图' };
    }
    try {
      const [ref] = await store.saveImages([
        { data: Buffer.from(image.data, 'base64'), mediaType: image.mediaType, name: 'screen.jpg' },
      ]);
      imageRef = ref;
    } catch (e) {
      return {
        ok: false,
        reason: 'generate-error',
        message: `截图保存失败：${e instanceof Error ? e.message : String(e)}`,
      };
    }
  }
  // 依次尝试：配置的模型失败（凭据被删 / 模型下架 / 该服务商没配 key）时**回落到当前对话的模型
  // 重试一次**，仍失败才抛出失败原因——最后一次是用户当前真正在用的模型，报错更有意义。
  // 兜底文案与失败分支同源，避免两处字面量漂移
  let last: ChatGenerateResult = { ok: false, reason: 'generate-error', message: noTextReason(undefined) };
  for (const [i, sel] of candidates.entries()) {
    const result = await generateWith(ctx, llm, sel, system, history, userText, pool, imageRef, memorySummary);
    if (result.ok) return result;
    last = result;
    // 回落要留痕：配的模型没生效时能在日志里看到原因（否则"我明明配了 A，怎么还是用 B"没法查）
    if (i + 1 < candidates.length) {
      console.warn(
        `dsh-pet: 对话用 ${modelLabel(sel)} 生成失败（${result.message ?? result.reason}），回落到 ${modelLabel(candidates[i + 1])}`,
      );
    }
  }
  return last;
}

/** 用**一个**确定的 provider/model 跑一次对话生成（候选链的一环；失败原样返回结构化原因，不吞） */
async function generateWith(
  ctx: { llm?: unknown },
  llm: { stream(o: unknown): AsyncIterable<unknown> },
  sel: ModelRef,
  system: string,
  history: ChatMemoryMessage[],
  userText: string,
  pool: MemeEntry[],
  imageRef?: unknown,
  memorySummary?: string,
): Promise<ChatGenerateResult> {
  // 历史 → dsh-llm 消息：user 经 createUserMessage（plugin 来源），assistant 经 createAssistantMessage
  const historyMessages = history.map((m) =>
    m.role === 'user'
      ? createUserMessage({
          content: [{ type: 'text', text: m.content }],
          source: { kind: 'user' },
        })
      : createAssistantMessage({
          content: [{ type: 'text', text: m.content }],
          source: { provider: sel.provider, model: sel.model },
        }),
  );

  const deadline = AbortSignal.timeout(TIMEOUT_MS);
  // 仅当模型声明支持 reasoning effort（含 "off"）时才传，否则省略：
  // 无 reasoning 元数据的模型（如 reasoningEfforts: false）显式传 off 会被
  // dsh-llm 判为 UNSUPPORTED_REASONING_EFFORT 并折叠成空流（表现为"模型未返回文本"）。
  const supportsOff = await supportsReasoningOff(ctx, sel.provider, sel.model);
  // 配图：池非空才把清单与指令附进 user 正文（空池 = 关闭配图，正文与旧行为逐字一致）
  const wantImage = pool.length > 0;
  // user 消息 content 按需拼：截图（image 块）在前、正文（含配图指令）在后。
  // 图片块必须在 content 数组里与文本并列，不能塞进某个 text 里——dsh-llm 按块类型路由。
  const userContent: ContentBlock[] = [];
  if (imageRef !== undefined) {
    // ImageAttachmentRef 由 dsh-attachment 拥有、dsh-llm 只 import 不 re-export，
    // 故这里从 ImageBlock 的字段派生该类型，避免直接名不存在的导入。
    userContent.push({ type: 'image', attachment: imageRef as ImageBlock['attachment'] });
  }
  userContent.push({ type: 'text', text: wantImage ? userText + imageInstruction(pool) : userText });
  // 远期记忆摘要拼进 system 尾部（不是 messages）：
  //   - 语义上它是「背景设定：很久以前发生过这些」，不是「刚说过的话」，放 system 才对
  //   - 放进 messages 会让模型以为那是一轮真实对话、甚至照着摘要续写回答
  // 无摘要时 system 逐字不变（与旧行为一致），故这里只在非空时拼接。
  const summary = memorySummary?.trim();
  const systemWithMemory =
    summary && summary.length > 0
      ? `${system}\n\n【很久之前的记忆摘要】以下是更早的对话的大致内容（未必逐字准确），供你了解来龙去脉，不要逐句复述：\n${summary}`
      : system;
  const options = {
    provider: sel.provider,
    model: sel.model,
    // 调用身份（见 llm-call.ts）：缺它时 opencode-go 回 400
    sessionId: PET_CALL_IDENTITY,
    messages: [
      ...historyMessages,
      createUserMessage({
        content: userContent,
        source: { kind: 'user' },
      }),
    ],
    system: systemWithMemory,
    // 不显式限 maxTokens：与碎碎念同因——推理模型把思考计入预算，显式小上限会截断正文；
    // 不传由 dsh-llm 按适配器 defaultMaxTokens 兜底，对话短回复输出仍短。
    temperature: 1,
    // 统一关闭深度思考：闲聊对话不需要推理（与碎碎念同一约束；仅模型声明支持时传）
    ...(supportsOff ? { reasoningEffort: ReasoningEffortId('off') } : {}),
    signal: deadline,
  };

  const assembler = new BlockAssembler();
  try {
    for await (const chunk of llm.stream(options)) {
      assembler.push(chunk as Parameters<BlockAssembler['push']>[0]);
    }
  } catch (e) {
    return {
      ok: false,
      reason: 'generate-error',
      message: e instanceof Error ? e.message : String(e),
    };
  }
  const text = assembler
    .blocks()
    .filter((b) => b.type === 'text')
    .map((b) => ('text' in b ? (b as { text: string }).text : ''))
    .join('')
    .trim();
  if (!text) return { ok: false, reason: 'generate-error', message: noTextReason(assembler.finish) };
  // 配图解析：命中池内才采纳并剥离标记；未选/幻觉名称 → 原样返回纯文本
  if (!wantImage) return { ok: true, text };
  const picked = extractChatImage(text, pool);
  return picked.image ? { ok: true, text: picked.text, image: picked.image } : { ok: true, text: picked.text };
}
