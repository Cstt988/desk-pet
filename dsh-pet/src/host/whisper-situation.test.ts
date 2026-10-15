/**
 * 碎碎念情境判断单元测试 —— 钉住「该不该冒泡」的规则优先级：
 *   1. 最短间隔是硬闸，优先级最高（刚出结果也不突破）
 *   2. 忙/出错 → 静默（除非超过「最长静默」上限）
 *   3. 刚出结果 → 说，celebrate（且不受深夜降频影响）
 *   4. 空闲 → 说；深夜要求 4 倍间隔
 * 以及 toneInstruction 的 casual → null（保证默认路径与改造前逐字一致）。
 *
 * 跑法：node --experimental-strip-types --test src/host/whisper-situation.test.ts
 */
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { decideWhisper, toneInstruction, type WhisperSituation } from './whisper-situation.ts';

/** 造一个"空闲、已过最短间隔"的基线情境，各用例只覆盖自己关心的字段 */
function base(over: Partial<WhisperSituation> = {}): WhisperSituation {
  return {
    workState: null,
    sinceLastMs: 600_000, // 10 分钟，超过默认 300s 的最短间隔
    minIntervalSec: 300,
    timeSlot: '下午',
    isWeekend: false,
    maxSilenceSec: 0, // 不设上限
    ...over,
  };
}

describe('decideWhisper', () => {
  test('空闲且已过最短间隔 → 说一句，casual', () => {
    const d = decideWhisper(base());
    assert.equal(d.speak, true);
    assert.equal(d.tone, 'casual');
  });

  test('未到最短间隔 → 不说（硬闸，最高优先级）', () => {
    const d = decideWhisper(base({ sinceLastMs: 100_000 })); // 100s < 300s
    assert.equal(d.speak, false);
    assert.match(d.reason, /未到最短间隔/);
  });

  test('最短间隔是硬闸：即便刚出结果也不突破', () => {
    const d = decideWhisper(base({ workState: 'success', sinceLastMs: 1_000 }));
    assert.equal(d.speak, false);
  });

  for (const state of ['thinking', 'working', 'waiting', 'error'] as const) {
    test(`忙碌/报错状态 ${state} → 静默不打扰`, () => {
      const d = decideWhisper(base({ workState: state }));
      assert.equal(d.speak, false);
      assert.match(d.reason, /不打扰/);
    });
  }

  for (const state of ['success', 'result'] as const) {
    test(`刚出结果状态 ${state} → 说一句，celebrate`, () => {
      const d = decideWhisper(base({ workState: state }));
      assert.equal(d.speak, true);
      assert.equal(d.tone, 'celebrate');
    });
  }

  test('深夜空闲但没到 4 倍间隔 → 降频不说', () => {
    // 深夜阈值 = 300 * 4 = 1200s；给 600s 应被拦
    const d = decideWhisper(base({ timeSlot: '深夜', sinceLastMs: 600_000 }));
    assert.equal(d.speak, false);
    assert.match(d.reason, /降频/);
  });

  test('深夜空闲且已过 4 倍间隔 → 说一句', () => {
    const d = decideWhisper(base({ timeSlot: '深夜', sinceLastMs: 1_300_000 }));
    assert.equal(d.speak, true);
  });

  test('凌晨与深夜同一套降频规则', () => {
    const d = decideWhisper(base({ timeSlot: '凌晨', sinceLastMs: 600_000 }));
    assert.equal(d.speak, false);
  });

  test('深夜出结果不受降频影响（结果时刻优先）', () => {
    const d = decideWhisper(base({ workState: 'success', timeSlot: '深夜', sinceLastMs: 400_000 }));
    // 400s > 300s 硬闸，虽未到深夜的 1200s，但出结果是最高价值时刻
    assert.equal(d.speak, true);
    assert.equal(d.tone, 'celebrate');
  });

  test('最长静默上限会打破"忙碌不打扰"', () => {
    const d = decideWhisper(
      base({ workState: 'working', sinceLastMs: 4 * 3600_000, maxSilenceSec: 3 * 3600 }),
    );
    assert.equal(d.speak, true);
    assert.match(d.reason, /最长静默/);
  });

  test('最长静默上限未到时不打破忙碌静默', () => {
    const d = decideWhisper(
      base({ workState: 'working', sinceLastMs: 1_000_000, maxSilenceSec: 3 * 3600 }),
    );
    assert.equal(d.speak, false);
  });

  test('最长静默上限也会打破深夜降频', () => {
    const d = decideWhisper(
      base({ timeSlot: '深夜', sinceLastMs: 4 * 3600_000, maxSilenceSec: 3 * 3600 }),
    );
    assert.equal(d.speak, true);
  });

  test('最长静默上限为 0 = 不设上限，忙碌仍静默', () => {
    const d = decideWhisper(
      base({ workState: 'working', sinceLastMs: 100 * 3600_000, maxSilenceSec: 0 }),
    );
    assert.equal(d.speak, false);
  });

  test('周末空闲照常说，reason 里带周末提示', () => {
    const d = decideWhisper(base({ isWeekend: true }));
    assert.equal(d.speak, true);
    assert.match(d.reason, /周末/);
  });

  test('minIntervalSec = 0 时不再拦（用户显式关掉频率下限）', () => {
    const d = decideWhisper(base({ minIntervalSec: 0, sinceLastMs: 0 }));
    assert.equal(d.speak, true);
  });

  test('从未说过（sinceLastMs = Infinity）总能说', () => {
    const d = decideWhisper(base({ sinceLastMs: Number.POSITIVE_INFINITY }));
    assert.equal(d.speak, true);
  });
});

describe('toneInstruction', () => {
  test('casual 返回 null（默认路径与改造前逐字一致）', () => {
    assert.equal(toneInstruction('casual'), null);
  });

  test('celebrate / comfort 各给一句语气要求', () => {
    assert.match(toneInstruction('celebrate') ?? '', /高兴/);
    assert.match(toneInstruction('comfort') ?? '', /关心/);
  });
});
