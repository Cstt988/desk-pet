/**
 * 记忆整理单元测试 —— 钉住「哪些消息会被压、哪些必须留、失败时怎么办」的取舍规则。
 *
 * 这是**有损**操作，所以边界必须逐条钉死：
 * - 切分只在「整轮」边界上（否则摘要与原文各留半截对话，读起来自相矛盾）；
 * - 没过期的消息、最近 keepRounds 轮，一条都不能压；
 * - 摘要生成失败 → 一个字都不许改（宁可暂时不整理，也不能删了原文却没摘要兜住）；
 * - 摘要累积：新摘要必须把旧摘要一起喂给模型，否则远期记忆会在下一次整理时丢失；
 * - 覆盖写重建（而非就地删），因此必须能在「无可压」时完全不动（changed=false）。
 *
 * 纯函数模块（不 import 宿主上下文、不碰网络），故本测试直接 import 它。
 *
 * 跑法：node --experimental-strip-types --test src/host/memory-compact.test.ts
 */
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

import type { ChatMemoryMessage } from './chat.ts';
import {
  buildCompactPrompt,
  compactPetMemory,
  normalizeSummary,
  splitForCompact,
  type CompactOptions,
} from './memory-compact.ts';

const DAY = 24 * 60 * 60 * 1000;

/** 造一条消息；ts 以 now 为基准往前推 days 天 */
function msg(role: 'user' | 'assistant', content: string, ts: number): ChatMemoryMessage {
  return { role, content, ts };
}

/** 造 n 轮（每轮 1 问 1 答）消息，第 i 轮的时间为 now - daysAgo[i] */
function rounds(n: number, now: number, daysAgo: (i: number) => number): ChatMemoryMessage[] {
  const out: ChatMemoryMessage[] = [];
  for (let i = 0; i < n; i++) {
    const ts = now - daysAgo(i) * DAY;
    out.push(msg('user', `问${i}`, ts));
    out.push(msg('assistant', `答${i}`, ts + 1000));
  }
  return out;
}

const NOW = 1_700_000_000_000;
const opts = (o: Partial<CompactOptions> = {}): CompactOptions => ({
  compactDays: 30,
  keepRounds: 2,
  maxChars: 800,
  now: NOW,
  ...o,
});

describe('splitForCompact —— 切分边界', () => {
  test('消息不足 keepRounds 轮 → 全部保留，无可压', () => {
    const msgs = rounds(2, NOW, () => 100); // 2 轮，keepRounds=2
    const { toCompact, keep } = splitForCompact(msgs, opts());
    assert.equal(toCompact.length, 0);
    assert.equal(keep.length, 4);
  });

  test('超期且在保留区之外的消息才被压', () => {
    // 6 轮：第 0-3 轮在 100 天前（超期），第 4-5 轮在 1 天前（未超期）；keepRounds=2
    const msgs = rounds(6, NOW, (i) => (i < 4 ? 100 : 1));
    const { toCompact, keep } = splitForCompact(msgs, opts());
    // 保留区 = 最近 2 轮（第 4、5 轮，共 4 条），本身就没超期；
    // 第 0-3 轮（8 条）全部超期且在保留区外 → 全部压
    assert.equal(toCompact.length, 8);
    assert.equal(keep.length, 4);
    assert.equal(toCompact[0].content, '问0');
    assert.equal(keep[0].content, '问4');
  });

  test('未超期但位于保留区外的消息**不压**（宁可留着也不丢）', () => {
    // 6 轮全在 1 天前（都没超期），keepRounds=2 → 保留区外有 4 轮，但都没超期
    const msgs = rounds(6, NOW, () => 1);
    const { toCompact, keep } = splitForCompact(msgs, opts());
    assert.equal(toCompact.length, 0);
    assert.equal(keep.length, 12);
  });

  test('超期消息按顺序处理，遇到第一个未超期的轮就停（后面的只会更新）', () => {
    // 第 0-1 轮超期、第 2 轮未超期、第 3 轮超期（时间倒挂的异常数据）
    const msgs = rounds(4, NOW, (i) => (i === 2 ? 1 : 100));
    const { toCompact } = splitForCompact(msgs, opts({ keepRounds: 0 }));
    // 第 0、1 轮超期 → 压进；第 2 轮未超期 → 停；第 3 轮即便超期也不再纳入
    assert.equal(toCompact.length, 4);
    assert.equal(toCompact[0].content, '问0');
    assert.equal(toCompact[3].content, '答1');
  });

  test('切分点落在整轮边界上（不会把一问一答拆开）', () => {
    const msgs = rounds(5, NOW, () => 100); // 全部超期
    const { toCompact, keep } = splitForCompact(msgs, opts({ keepRounds: 2 }));
    assert.equal(toCompact.length % 2, 0, '压入部分必须是整数轮');
    assert.equal(keep.length % 2, 0, '保留部分必须是整数轮');
    assert.equal(toCompact.length + keep.length, msgs.length, '不许丢消息也不许凭空多');
    // 问答成对：压入区必须首条为 user、末条为 assistant
    assert.equal(toCompact[0].role, 'user');
    assert.equal(toCompact[toCompact.length - 1].role, 'assistant');
  });

  test('keepRounds = 0 → 保留区为空，全部超期消息都可压', () => {
    const msgs = rounds(3, NOW, () => 100);
    const { toCompact, keep } = splitForCompact(msgs, opts({ keepRounds: 0 }));
    assert.equal(toCompact.length, 6);
    assert.equal(keep.length, 0);
  });

  test('恰好等于 compactDays 边界的消息不压（用严格早于判断）', () => {
    // 第 0 轮恰好 30 天前（= cutoff），第 1-2 轮在 1 天前
    const msgs = rounds(3, NOW, (i) => (i === 0 ? 30 : 1));
    const { toCompact } = splitForCompact(msgs, opts({ keepRounds: 0 }));
    // ts < cutoff 为假（相等不满足严格小于）→ 不压
    assert.equal(toCompact.length, 0);
  });
});

describe('compactPetMemory —— 重建行为', () => {
  test('无可压消息 → changed=false 且返回**同一个对象**（不做无谓结构改写）', async () => {
    const memory = { messages: rounds(2, NOW, () => 1) };
    const r = await compactPetMemory(memory, opts(), async () => ({ ok: true, summary: '不该被调用' }));
    assert.equal(r.changed, false);
    assert.equal(r.compactedCount, 0);
    assert.equal(r.memory, memory, '必须原样返回同一引用');
  });

  test('有可压消息 → 保留区留下、摘要写入、消息数减少', async () => {
    const memory = { messages: rounds(6, NOW, (i) => (i < 4 ? 100 : 1)) };
    const r = await compactPetMemory(memory, opts(), async () => ({ ok: true, summary: '早年的概括' }));
    assert.equal(r.changed, true);
    assert.equal(r.compactedCount, 8);
    assert.equal(r.memory.messages.length, 4);
    assert.equal(r.memory.summary, '早年的概括');
  });

  test('摘要生成失败 → 一个字都不改（宁可不整理，也不能删了原文却没摘要）', async () => {
    const memory = { messages: rounds(6, NOW, (i) => (i < 4 ? 100 : 1)) };
    const r = await compactPetMemory(memory, opts(), async () => ({ ok: false, reason: 'generate-error' }));
    assert.equal(r.changed, false);
    assert.equal(r.compactedCount, 0);
    assert.equal(r.memory, memory);
    assert.equal(r.memory.messages.length, 12, '原文必须一条不少地留着');
  });

  test('模型返回空白摘要 → 不写盘（空摘要等于把原文删了又什么都没换回来）', async () => {
    const memory = { messages: rounds(6, NOW, (i) => (i < 4 ? 100 : 1)) };
    const r = await compactPetMemory(memory, opts(), async () => ({ ok: true, summary: '   \n\n  ' }));
    assert.equal(r.changed, false);
    assert.equal(r.memory.messages.length, 12);
  });

  test('不修改传入的 memory 对象（纯函数）', async () => {
    const original = rounds(6, NOW, (i) => (i < 4 ? 100 : 1));
    const memory = { messages: original };
    await compactPetMemory(memory, opts(), async () => ({ ok: true, summary: '概括' }));
    assert.equal(memory.messages, original, '原对象引用不得被替换');
    assert.equal(original.length, 12, '原数组内容不得被就地改动');
  });

  test('把 prompt 透传给生成器（调用方据此接模型）', async () => {
    const memory = { messages: rounds(6, NOW, (i) => (i < 4 ? 100 : 1)) };
    let seen = '';
    await compactPetMemory(memory, opts(), async (p) => {
      seen = p;
      return { ok: true, summary: '概括' };
    });
    assert.match(seen, /问0/);
    assert.match(seen, /用户：/);
  });
});

describe('buildCompactPrompt —— 摘要累积', () => {
  const msgs = [msg('user', '我喜欢猫', 1), msg('assistant', '喵～', 2)];

  test('无旧摘要 → 只有「需要总结的对话」段', () => {
    const p = buildCompactPrompt(undefined, msgs);
    assert.match(p, /【需要总结的对话】/);
    assert.ok(!p.includes('已有的远期记忆摘要'));
  });

  test('有旧摘要 → 一并喂给模型（否则远期记忆会在下次整理时丢掉）', () => {
    const p = buildCompactPrompt('用户养了一只叫团子的猫。', msgs);
    assert.match(p, /【已有的远期记忆摘要】/);
    assert.match(p, /团子/);
    assert.match(p, /【需要并入摘要的新对话】/);
  });

  test('空白旧摘要视同没有（不产生空段落）', () => {
    const p = buildCompactPrompt('   \n ', msgs);
    assert.ok(!p.includes('已有的远期记忆摘要'));
  });

  test('带角色前缀，模型才分得清谁说的', () => {
    const p = buildCompactPrompt(undefined, msgs);
    assert.match(p, /用户：我喜欢猫/);
    assert.match(p, /桌宠：喵～/);
  });
});

describe('normalizeSummary —— 收尾与截断', () => {
  test('折叠多余空行并去首尾空白', () => {
    assert.equal(normalizeSummary('\n\n甲\n\n\n\n乙\n\n', 100), '甲\n\n乙');
  });

  test('CRLF 归一化为 LF', () => {
    assert.equal(normalizeSummary('甲\r\n乙', 100), '甲\n乙');
  });

  test('超长按上限截断并加省略号', () => {
    const s = normalizeSummary('一'.repeat(20), 10);
    assert.equal(s.length, 11); // 10 字 + 省略号
    assert.equal(s, '一'.repeat(10) + '…');
  });

  test('正好等于上限 → 不截断、不加省略号', () => {
    assert.equal(normalizeSummary('一'.repeat(10), 10), '一'.repeat(10));
  });

  test('超出 1 个字符 → 截断', () => {
    assert.equal(normalizeSummary('一'.repeat(11), 10), '一'.repeat(10) + '…');
  });

  test('上限为 0 / 负数也至少留 1 字（不允许产出空摘要）', () => {
    assert.equal(normalizeSummary('甲', 0), '甲');
  });
});

describe('端到端：两次整理累积（C 方案核心）', () => {
  test('第二次整理把旧摘要带进 prompt，且新摘要覆盖旧摘要', async () => {
    // 第一次：4 轮超期 + 2 轮近期
    const first = { messages: rounds(6, NOW, (i) => (i < 4 ? 100 : 1)) };
    const r1 = await compactPetMemory(first, opts(), async () => ({ ok: true, summary: '第一次摘要' }));
    assert.equal(r1.memory.summary, '第一次摘要');
    assert.equal(r1.memory.messages.length, 4);

    // 第二次：把第一次保留的 4 条也变成超期（模拟时间流逝），再加 2 轮新的近期对话
    const later = NOW + 40 * DAY;
    const carried = r1.memory.messages.map((m) => ({ ...m, ts: m.ts - 100 * DAY }));
    const second = { messages: [...carried, ...rounds(2, later, () => 1)], summary: r1.memory.summary };
    let prompt = '';
    const r2 = await compactPetMemory(
      second,
      opts({ now: later, keepRounds: 2 }),
      async (p) => {
        prompt = p;
        return { ok: true, summary: '第二次摘要（已含第一次）' };
      },
    );
    assert.match(prompt, /第一次摘要/, '旧摘要必须进 prompt —— 否则远期记忆丢失');
    assert.equal(r2.memory.summary, '第二次摘要（已含第一次）', '新摘要覆盖旧摘要');
    assert.equal(r2.memory.messages.length, 4, '只剩最近 2 轮原文');
  });
});
