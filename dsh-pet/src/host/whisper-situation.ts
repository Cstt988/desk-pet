/**
 * 碎碎念的**情境判断**（纯函数，可单测）：决定"现在这一刻该不该冒泡、冒泡该说什么调子"。
 *
 * 背景（改造动机）：原行为是纯定时器——到点必说一句，不管用户当下在干什么。
 * 结果是：正忙得不可开交时桌宠插科打诨、深夜还在那儿冒泡、刚出结果时它反而沉默，
 * 全都"卡在钟点上看不出场合"。本模块把"该不该说"从时间里摘出来，交给情境：
 *
 *   - **忙的时候别打扰**：thinking / working / waiting → 静默（用户在等结果，插话是噪音）
 *   - **刚出结果时说一句**：success / result → 值得说（这是最有价值的一刻）
 *   - **出错时别嬉皮笑脸**：error → 静默（用户正烦，说俏皮话是负分）
 *   - **深夜降频**：凌晨时段 → 想说也得看时机
 *
 * 设计原则：
 * - **纯函数**：不读时钟、不读文件、不调模型。所有输入由调用方传进来，因此可以完整单测；
 *   时钟不在这里读（与 time-context.ts 同一约定），避免测试要 mock 时间。
 * - **只判断，不生成**：本模块不碰 LLM。判断"该不该"是本地规则，成本为零；
 *   一旦为了判断去调模型，"判断"就比"生成"还贵 —— 这是刻意的成本取舍。
 * - **可解释**：每次判断都返回 reason 字符串，日志里能看出"为什么这次没说"，
 *   否则用户只会看到"桌宠不说话了"却不知道为什么，无法调参。
 */

import type { HostWorkStatusState } from './work-status';

/** 说话的语气/内容倾向：驱动 prompt 的选择（同一次生成只用一个） */
export type WhisperTone = 'casual' | 'celebrate' | 'comfort';

/** 判断输入：全部由调用方采集（本模块不读时钟/文件） */
export interface WhisperSituation {
  /** 工作状态快照的 state（null = 尚无会话活动，视为空闲） */
  workState: HostWorkStatusState | null;
  /** 距上次碎碎念的毫秒数（Infinity = 从未说过） */
  sinceLastMs: number;
  /** 用户配置的最短间隔下限（秒）：即便情境再好也不比这个更频繁 */
  minIntervalSec: number;
  /** 本地小时的时段（由 time-context 的时段判定传入，'深夜'/'凌晨' 等） */
  timeSlot: string;
  /** 是否工作日（周末节奏更松） */
  isWeekend: boolean;
  /** 用户配置的"最长静默"上限（秒）：0 = 不设上限 */
  maxSilenceSec: number;
}

/** 判断结果 */
export interface WhisperDecision {
  /** 该不该冒泡 */
  speak: boolean;
  /** 该用什么调子（speak 为 false 时无意义） */
  tone: WhisperTone;
  /** 可读的原因（写日志用；也便于用户理解为何沉默） */
  reason: string;
}

/** 深夜时段名（与 time-context 的 SLOTS 命名一致）：这两个时段整体降频 */
const LATE_NIGHT_SLOTS = new Set(['深夜', '凌晨']);

/** 忙/不宜打扰的状态：这些状态下静默（用户在等结果或正烦着） */
const BUSY_STATES = new Set<HostWorkStatusState>(['thinking', 'working', 'waiting', 'error']);

/** 值得说话的"刚出结果"状态 */
const RESULT_STATES = new Set<HostWorkStatusState>(['success', 'result']);

/**
 * 深夜的间隔倍增：深夜能说，但要求"距上次"至少是这个倍数。
 * 取 4：默认 300s 的最短间隔在深夜变成 1200s（20 分钟），既不彻底沉默也不频繁。
 */
const LATE_NIGHT_INTERVAL_FACTOR = 4;

/**
 * 判断此刻该不该冒泡。**规则按优先级从高到低**（先命中的先返回）：
 *
 * 1. 未到最短间隔 → 不说（这是硬闸，任何情境都不能突破用户设的频率下限）
 * 2. 忙 / 出错 → 不说（除非已超过"最长静默"上限：用户明确要求再久也要说一句）
 * 3. 刚出结果 → 说，调子 celebrate（最高价值时刻，不受深夜降频影响）
 * 4. 其余（空闲/无会话活动）→ 说，调子 casual；深夜要把间隔乘 4 才说
 *
 * 第 2 条的例外很重要：用户设了 maxSilenceSec（比如 3 小时）就是"宁可打扰也别一直沉默"，
 * 这时即便在忙也要说一句 —— 否则长时间干活时桌宠会像死了一样。
 */
export function decideWhisper(situation: WhisperSituation): WhisperDecision {
  const {
    workState,
    sinceLastMs,
    minIntervalSec,
    timeSlot,
    isWeekend,
    maxSilenceSec,
  } = situation;

  // 1) 硬闸：最短间隔。maxSilenceSec 在下面单独开例外，不在这里
  const minMs = minIntervalSec * 1000;
  if (sinceLastMs < minMs) {
    return {
      speak: false,
      tone: 'casual',
      reason: `未到最短间隔（距上次 ${Math.round(sinceLastMs / 1000)}s < ${minIntervalSec}s）`,
    };
  }

  // 「最长静默」例外：超过上限就无视忙/深夜，说一句（用户显式要求别一直沉默）
  const overridden = maxSilenceSec > 0 && sinceLastMs >= maxSilenceSec * 1000;

  // 2) 忙 / 出错 → 静默（除非已超过最长静默上限）
  if (workState !== null && BUSY_STATES.has(workState)) {
    if (!overridden) {
      return { speak: false, tone: 'casual', reason: `当前${stateLabel(workState)}，不打扰` };
    }
    // 静默太久，用户显式要求"宁可打扰也别一直沉默"：这时说一句安慰/打气的调子
    return {
      speak: true,
      tone: workState === 'error' ? 'comfort' : 'casual',
      reason: `当前${stateLabel(workState)}，但已超过最长静默，补一句`,
    };
  }

  // 3) 刚出结果 → 说（这是最值得说话的时刻，深夜也不降频）
  if (workState !== null && RESULT_STATES.has(workState)) {
    return {
      speak: true,
      tone: 'celebrate',
      reason: overridden ? '刚出结果（超过最长静默，补一句）' : '刚出结果，值得说一句',
    };
  }

  // 4) 空闲 / 无会话活动 → 说；深夜要求更长的间隔（放宽，但不沉默）
  const late = LATE_NIGHT_SLOTS.has(timeSlot);
  if (late && !overridden) {
    const lateMinMs = minMs * LATE_NIGHT_INTERVAL_FACTOR;
    if (sinceLastMs < lateMinMs) {
      return {
        speak: false,
        tone: 'casual',
        reason: `${timeSlot}降频（距上次 ${Math.round(sinceLastMs / 1000)}s < ${Math.round(lateMinMs / 1000)}s）`,
      };
    }
  }

  // 周末更松：语气提示带上，让模型自己挑更懒散的说法（不改变"说不说"）
  const weekendNote = isWeekend ? '（周末）' : '';
  return {
    speak: true,
    tone: 'casual',
    reason: late ? `${timeSlot}时段（已过降频阈值）` : `空闲${weekendNote}，随口说一句`,
  };
}

/** 工作状态的中文名（日志可读性；与前端 events.workStatusTexts 各管一侧，这里只用于日志） */
function stateLabel(state: HostWorkStatusState): string {
  switch (state) {
    case 'thinking':
      return '思考中';
    case 'working':
      return '执行中';
    case 'waiting':
      return '等待中';
    case 'error':
      return '报错中';
    case 'success':
      return '刚完成';
    case 'result':
      return '出结果';
  }
}

/**
 * 按调子给碎碎念指令补一句语气要求。
 *
 * 为什么由这里给（而不是只用人设 whisperPrompt）：人设决定"是谁在说"，调子决定
 * "这一刻该用什么口吻"——刚出结果时说"恭喜完成"和深夜说"还不睡呀"是两种话，
 * 纯人设给不了这个区分。**返回 null 表示不需要额外要求**（casual 走原指令），
 * 这样默认路径与改造前逐字一致，零回归风险。
 */
export function toneInstruction(tone: WhisperTone): string | null {
  switch (tone) {
    case 'celebrate':
      return '刚才那件事做完了/出结果了，为这件事随口高兴一句（别问是什么事，你不知道细节）。';
    case 'comfort':
      return '刚才出了点问题，用轻松的语气关心一句（别说教、别分析问题）。';
    case 'casual':
      return null;
  }
}
