/**
 * 记忆整理（C 方案：摘要累积）。
 *
 * 【要解决什么】`memory.json` 此前是**只增不减**的：`chatMemoryRounds` 只决定
 * 「带多少历史进上下文」，对磁盘文件大小没有任何作用。聊得越多文件越大，而
 * `readMemory`/`writeMemory` 每次都要 JSON.parse/stringify **整份文件**——
 * 文件越大，每次对话的延迟越明显；损坏时的代价也随体积放大（损坏 = 备份 + 从头开始）。
 *
 * 【怎么做】不「就地删除旧消息」（那是边读边改，中途崩了可能留下半截文件），
 * 而是**算出一份新记忆 → 整份覆盖写回**：
 *   1. 最近 keepRounds 轮原文 → 原样保留（细节不丢，这是用户最常回味的近事）
 *   2. 比 keepRounds 更早、且已超过 compactDays 的消息 → 压成一段「大体意思」摘要
 *   3. 已有摘要与新压出来的内容**累积合并**成新摘要（关键！每次只压增量，
 *      所以压缩成本不随历史增长——这是 C 方案相对「每次重压全部历史」的本质优势）
 *
 * 【为什么摘要要累积而不是重算】把「旧摘要 + 这中间新增的旧消息」交给模型产出新摘要，
 * 输入长度恒定为「一段摘要 + 一批待压消息」，与总历史长度无关。若每次都重压全部原文，
 * 聊天记录上万条后单次压缩的 token 就会爆掉，整理本身变成新的问题。
 *
 * 【数据形态】整理只碰消息数组与摘要字段，不关心桶/实例的双层结构——那是 index.ts 的事。
 * 这样本模块可以脱离宿主上下文单独测试（见 memory-compact.test.ts）。
 */

import type { ChatMemoryMessage } from './chat';

/** 记忆整理的参数（已由配置层校验并填默认；这里只消费，不再校验） */
export interface CompactOptions {
  /** 超过这么多天的消息才可能被压（最近 keepRounds 轮永远豁免） */
  compactDays: number;
  /** 永远保留原文的最近轮数（1 轮 = 1 问 1 答） */
  keepRounds: number;
  /** 摘要正文上限字符数（超出截断，防模型啰嗦把摘要本身撑爆） */
  maxChars: number;
  /** 「现在」的时间戳（毫秒）；显式传入而不是内部取 Date.now()，测试才能确定性地跑 */
  now: number;
}

/** 一个实例的记忆（消息 + 可选摘要）。摘要字段是本次新增，旧文件没有即为 undefined */
export interface PetMemory {
  messages: ChatMemoryMessage[];
  /** 「很久之前」的大体意思；无摘要时字段不存在（而不是空串——省磁盘也更易判别） */
  summary?: string;
}

/** 整理结果：新记忆 + 是否真的需要写盘（不需要时调用方应跳过写，避免无谓的 IO 与 mtime 变动） */
export interface CompactResult {
  memory: PetMemory;
  changed: boolean;
  /** 本次被压进摘要的消息条数（0 = 无需整理） */
  compactedCount: number;
}

/**
 * 把一条消息渲染成给摘要模型看的一行。
 * 带角色前缀（「用户」「桌宠」）——否则模型分不清哪句是谁说的，摘要会串味。
 */
function renderMessage(m: ChatMemoryMessage): string {
  const who = m.role === 'user' ? '用户' : '桌宠';
  return `${who}：${m.content}`;
}

/**
 * 算出一个实例该压哪些消息、保留哪些。
 *
 * 切分点定义（刻意做成纯函数，测试直接断言）：
 *   - 轮（round）按「消息对」计：索引 floor(i / 2) 即第 i 条所属的轮次
 *   - 保留区起点 = max(0, 总数 - keepRounds * 2)，即最近 keepRounds 轮的原文
 *   - 候选区 = 保留区之前的全部消息，再按 ts 过滤出「早于 now - compactDays 天」的
 *   - 边界对齐：压进摘要的部分必须以「整轮」为界，否则会出现「用户问了、助理的答还在保留区」
 *     这种半截对话，摘要和原文会各留一半，读起来自相矛盾
 */
export function splitForCompact(
  messages: ChatMemoryMessage[],
  opts: CompactOptions,
): { toCompact: ChatMemoryMessage[]; keep: ChatMemoryMessage[] } {
  const total = messages.length;
  // 最近 keepRounds 轮原文的起点（keepRounds = 0 时即 0，全部消息都是候选）
  const keepStart = Math.max(0, total - Math.max(0, Math.floor(opts.keepRounds)) * 2);
  if (keepStart <= 0) return { toCompact: [], keep: messages.slice() };

  // 候选区里「已足够旧」的部分：以每轮**后者**（assistant 的 ts，通常更晚）判断该轮是否过期。
  // 用「整轮是否都过期」而不是「逐条判断」——逐条会把同一轮的两条拆开。
  const cutoff = opts.now - Math.max(0, opts.compactDays) * 24 * 60 * 60 * 1000;
  let compactEnd = 0; // 候选中「整轮都够旧」的结束位置（左闭右开）
  for (let start = 0; start + 1 < keepStart; start += 2) {
    const a = messages[start];
    const b = messages[start + 1];
    // 该轮的结束时间取两条里较晚的（缺一条时按另一条）
    const roundEndTs = b ? Math.max(a.ts, b.ts) : a.ts;
    if (roundEndTs < cutoff) compactEnd = start + 2;
    else break; // 消息按时间正序，一旦遇到没过期的轮，后面的只会更新
  }
  // 奇数长度消息的兜底：单出来的最后一条若不与任何轮成对，不参与压缩（宁可留着）
  return { toCompact: messages.slice(0, compactEnd), keep: messages.slice(compactEnd) };
}

/**
 * 把待压消息渲染成给模型看的文本（含已有摘要，让它做「累积合并」而不是重写）。
 *
 * 【为什么要把旧摘要也给模型】这正是 C 方案的核心：新摘要 = f(旧摘要, 本次待压消息)。
 * 不给旧摘要，模型就会「只总结这一段」，之前压进去的远期记忆会在下一次整理时丢掉。
 */
export function buildCompactPrompt(
  previousSummary: string | undefined,
  toCompact: ChatMemoryMessage[],
): string {
  const lines: string[] = [];
  if (previousSummary && previousSummary.trim().length > 0) {
    lines.push('【已有的远期记忆摘要】');
    lines.push(previousSummary.trim());
    lines.push('');
    lines.push('【需要并入摘要的新对话】');
  } else {
    lines.push('【需要总结的对话】');
  }
  lines.push(toCompact.map(renderMessage).join('\n'));
  return lines.join('\n');
}

/** 摘要的产出形态：失败时不抛，交给调用方决定是否写盘（整理失败绝不能影响正常对话） */
export type CompactSummaryResult =
  | { ok: true; summary: string }
  | { ok: false; reason: 'generate-error'; message?: string };

/**
 * 收尾：规范化模型给出的摘要文本。
 *   - 折叠多余空行、去首尾空白
 *   - 超长按 maxChars 截断（并在末尾加省略号，让读者知道被截了）
 * 独立成函数便于测试边界（正好等于上限、超出 1 个字符、全空白）。
 */
export function normalizeSummary(raw: string, maxChars: number): string {
  const text = raw
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  const limit = Math.max(1, Math.floor(maxChars));
  if (text.length <= limit) return text;
  return text.slice(0, limit) + '…';
}

/**
 * 对单个实例执行一次整理（纯函数版本：摘要生成由调用方注入）。
 *
 * 【为什么把「调模型」抽成回调】本模块要能脱离宿主上下文做单元测试——把 summary 的
 * 生成过程注入进来，测试就能用「固定返回一段文字」的假实现覆盖全部边界，不起网络。
 *
 * @param memory 该实例现有记忆（原对象不会被修改——纯函数，调用方拿到的是新对象）
 * @param opts 整理参数
 * @param generate 摘要生成器：接收给模型的提示词，返回摘要文本（失败时返回 ok:false）
 */
export async function compactPetMemory(
  memory: PetMemory,
  opts: CompactOptions,
  generate: (prompt: string) => Promise<CompactSummaryResult>,
): Promise<CompactResult> {
  const { toCompact, keep } = splitForCompact(memory.messages ?? [], opts);
  if (toCompact.length === 0) {
    // 没有可压的：连 messages 字段都原样保留（不做无谓的结构改写）
    return { memory, changed: false, compactedCount: 0 };
  }
  const prompt = buildCompactPrompt(memory.summary, toCompact);
  const generated = await generate(prompt);
  if (!generated.ok) {
    // 生成失败 → 不改动任何东西。宁可暂时不整理，也不能把原文删掉却没有摘要兜住
    // （那等于静默丢记忆，比不整理严重得多）。
    return { memory, changed: false, compactedCount: 0 };
  }
  const summary = normalizeSummary(generated.summary, opts.maxChars);
  if (summary.length === 0) {
    return { memory, changed: false, compactedCount: 0 };
  }
  return {
    memory: { messages: keep, summary },
    changed: true,
    compactedCount: toCompact.length,
  };
}

/** 配置层把 compactDays / keepRounds / maxChars 的默认值集中在这里，避免散落在多处字面量 */
export const COMPACT_DEFAULTS = {
  /** 30 天前的对话算「很久之前」 */
  compactDays: 30,
  /** 最近 10 轮原文永远保留 */
  keepRounds: 10,
  /** 摘要正文最多 800 字符 */
  maxChars: 800,
  /** 默认每 6 小时整理一次 */
  periodSec: 6 * 60 * 60,
} as const;
