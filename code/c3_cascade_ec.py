# -*- coding: utf-8 -*-
"""
C3 级联语音翻译系统 —— 新增ASR识别文本纠错进阶功能
范式：ASR (Whisper-small) → LLM文本纠错 → LLM翻译 的级联，语音翻译(英文语音→中文)

进阶实现：ASR输出英文先经过大模型纠错，修正拼写、标点、识别错词，再送入翻译环节
覆盖基础+进阶要求：
  1) 实现 ASR → 文本纠错LLM → 翻译LLM 三级级联系统
  2) 支持英文语音 → 中文文本翻译
  3) 保存原始ASR文本、纠错后文本、最终中文翻译
  4) 完整统计ASR、纠错、翻译三阶段耗时
  5) 输出带纠错字段的cascade_results.json，兼容C4数据集id

运行命令不变，直接执行即可：
  python c3_cascade.py
离线本地模型运行：
  python c3_cascade.py --offline
"""

import os, sys, json, time, argparse

# 国内默认走 hf-mirror 镜像下载模型（必须在 import transformers 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 离线/本地模型支持：加 --offline，或把 --asr_model/--llm 指向本地模型目录，就不联网直接用本地/缓存模型。
def _maybe_offline():
    argv = sys.argv
    off = "--offline" in argv
    for flag in ("--model", "--asr_model", "--llm"):
        if flag in argv:
            k = argv.index(flag) + 1
            if k < len(argv) and os.path.isdir(argv[k]):
                off = True
    if off:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
_maybe_offline()

import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(HERE, "..", "..", "common_data", "dataset.json"))
    ap.add_argument("--asr_model", default="openai/whisper-small")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--start", type=int, default=0, help="起始下标(默认0)")
    ap.add_argument("--n", type=int, default=0, help="只跑前几条；默认 0 = 跑数据集全部")
    ap.add_argument("--offline", action="store_true", help="离线模式：不联网，用本地/缓存已下载的模型")
    ap.add_argument("--outdir", default=os.path.join(HERE, "..", "outputs"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ds = json.load(open(args.dataset, encoding="utf-8"))
    data_root = os.path.dirname(os.path.abspath(args.dataset))
    samples = ds[args.start:] if args.n <= 0 else ds[args.start:args.start + args.n]
    audio_paths = [os.path.join(data_root, s["audio"]) for s in samples]
    print(f"数据集: {os.path.abspath(args.dataset)}  （共 {len(ds)} 条，本次使用 {len(samples)} 条")
    device = 0 if torch.cuda.is_available() else -1

    # ===== 第一级：ASR（Whisper-small）语音 → 原始英文文本 =====
    print("=" * 60)
    print("第一级 ASR：Whisper-small  语音 → 原始识别英文文本")
    print("=" * 60)
    t0 = time.time()
    asr = pipeline("automatic-speech-recognition", model=args.asr_model,
                   torch_dtype=torch.float16 if device == 0 else torch.float32, device=device)
    asr_out = asr(audio_paths, batch_size=16,
                  generate_kwargs={"language": "english", "task": "transcribe"})
    raw_asr_texts = [o["text"].strip() for o in asr_out]
    asr_time = time.time() - t0
    print(f"ASR 识别完成，{len(samples)} 条，耗时 {asr_time:.1f}s")

    # ===== 加载LLM模型（纠错+翻译共用同一个模型） =====
    print("\n" + "=" * 60)
    print("加载通用LLM模型，用于文本纠错与翻译")
    print("=" * 60)
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm, torch_dtype=torch.float16 if device == 0 else torch.float32)
    llm = llm.to("cuda:0" if device == 0 else "cpu")
    print(f"LLM模型加载完成，耗时 {time.time()-t_load:.1f}s")

    # ===== 新增进阶模块：ASR识别文本纠错 =====
    print("\n" + "=" * 60)
    print("进阶模块：LLM 英文识别文本纠错")
    print("=" * 60)
    t0 = time.time()
    fixed_asr_texts = []
    for txt in raw_asr_texts:
        # 纠错专用提示词
        messages = [
            {"role": "system", "content": "你是英文文本校对员，只修正语音识别产生的拼写、标点、词汇错误。"},
            {"role": "user", "content": f"修正下面句子的识别错误，只输出修正后的完整英文，不要多余解释：\n{txt}"}
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            gen = llm.generate(**inputs, max_new_tokens=64, do_sample=False)
        fixed_txt = tok.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        fixed_asr_texts.append(fixed_txt)
    fix_time = time.time() - t0
    print(f"文本纠错完成，{len(samples)} 条，耗时 {fix_time:.1f}s")

    # ===== 第二级：LLM 纠错后英文 → 中文翻译 =====
    print("\n" + "=" * 60)
    print("第二级 LLM：纠错英文文本 → 中文翻译")
    print("=" * 60)
    t0 = time.time()
    translations = []
    for txt in fixed_asr_texts:
        messages = [
            {"role": "system", "content": "You are a professional English-to-Chinese translator."},
            {"role": "user", "content": f"Translate the following English sentence into Chinese. "
                                         f"Output ONLY the Chinese translation, no explanation.\n\n{txt}"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            gen = llm.generate(**inputs, max_new_tokens=64, do_sample=False)
        resp = tok.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        translations.append(resp)
    llm_time = time.time() - t0
    print(f"翻译完成，{len(samples)} 条，耗时 {llm_time:.1f}s")

    # ===== 保存结果：新增原始ASR、纠错后文本字段 =====
    results = []
    for s, raw_asr, fixed_asr, tr in zip(samples, raw_asr_texts, fixed_asr_texts, translations):
        results.append({
            "id": s["id"],
            "audio": s["audio"],
            "reference_en": s["text"],        # 数据集标准英文原文
            "asr_text_raw": raw_asr,           # ASR原始未纠错识别文本
            "asr_text_fixed": fixed_asr,       # LLM纠错后的标准英文文本
            "translation_zh": tr,              # 最终中文翻译
        })
    out_json = os.path.join(args.outdir, "cascade_results_ec.json")
    json.dump(results, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 汇总全流程耗时写入summary
    total_all_time = asr_time + fix_time + llm_time
    summary = {
        "asr_model": args.asr_model,
        "llm": args.llm,
        "num_samples": len(samples),
        "asr_time_sec": round(asr_time, 1),
        "text_fix_time_sec": round(fix_time, 1),
        "llm_trans_time_sec": round(llm_time, 1),
        "total_time_sec": round(total_all_time, 1)
    }
    json.dump(summary, open(os.path.join(args.outdir, "c3_summary_ec.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ===== 打印样例，直观展示纠错效果 =====
    print("\n---- 级联完整流程样例（原始ASR → 纠错后英文 → 中文翻译）----")
    for r in results[:6]:
        print(f"[{r['id']}]")
        print(f"  ① 原始ASR识别: {r['asr_text_raw']}")
        print(f"  ② LLM纠错文本: {r['asr_text_fixed']}")
        print(f"  ③ 最终中文翻译: {r['translation_zh']}")
    print(f"\n结果文件已保存: {out_json}")
    print(f"全流程总耗时: {total_all_time:.1f}s")
    print(f"分段耗时：ASR {asr_time:.1f}s | 文本纠错 {fix_time:.1f}s | 翻译 {llm_time:.1f}s")


if __name__ == "__main__":
    main()