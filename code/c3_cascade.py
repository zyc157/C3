# -*- coding: utf-8 -*-
"""
C3 级联语音翻译系统 —— 基础要求一键演示脚本
范式：ASR (Whisper-small) → LLM (Qwen2.5-1.5B-Instruct) 的级联，做"语音翻译"(英文语音→中文)。

覆盖基础要求：
  1) 实现 ASR → LLM 级联系统
  2) 支持一个语音翻译方向（英文语音 → 中文文本）
  3) 保存中间结果（ASR 英文文本）和最终输出（中文翻译）
  4) 说明级联范式的流程
  5) 与 C4 端到端系统使用同一批音频对比（输出 cascade_results.json，C4 复用同样的 id）

运行（默认对 common_data/dataset.json 里的全部音频做级联翻译，无需任何参数）：
  python c3_cascade.py

换用别的数据集（格式同 common_data/dataset.json：列表，每条含 "audio"(相对路径) 和 "text"(转录文本)）：
  python c3_cascade.py --dataset /你的路径/your_dataset.json

没网 / 镜像超时 / 已有本地模型时，加 --offline 用缓存模型：
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
    print(f"数据集: {os.path.abspath(args.dataset)}  （共 {len(ds)} 条，本次使用 {len(samples)} 条）")
    device = 0 if torch.cuda.is_available() else -1

    # ===== 第一级：ASR（Whisper-small）语音 → 英文文本 =====
    print("=" * 60)
    print("第一级 ASR：Whisper-small  语音 → 英文文本")
    print("=" * 60)
    t0 = time.time()
    asr = pipeline("automatic-speech-recognition", model=args.asr_model,
                   torch_dtype=torch.float16 if device == 0 else torch.float32, device=device)
    asr_out = asr(audio_paths, batch_size=16,
                  generate_kwargs={"language": "english", "task": "transcribe"})
    asr_texts = [o["text"].strip() for o in asr_out]
    asr_time = time.time() - t0
    print(f"ASR 完成，{len(samples)} 条，耗时 {asr_time:.1f}s")

    # ===== 第二级：LLM（Qwen2.5-1.5B）英文文本 → 中文翻译 =====
    print("\n" + "=" * 60)
    print("第二级 LLM：Qwen2.5-1.5B-Instruct  英文文本 → 中文翻译")
    print("=" * 60)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm, torch_dtype=torch.float16 if device == 0 else torch.float32)
    llm = llm.to("cuda:0" if device == 0 else "cpu")
    translations = []
    for txt in asr_texts:
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
    print(f"LLM 翻译完成，{len(samples)} 条，耗时 {llm_time:.1f}s")

    # ===== 保存中间结果 + 最终输出 =====
    results = []
    for s, asr_t, tr in zip(samples, asr_texts, translations):
        results.append({
            "id": s["id"], "audio": s["audio"],
            "reference_en": s["text"],     # 标准英文文本
            "asr_text": asr_t,             # 中间结果：ASR 识别的英文
            "translation_zh": tr,          # 最终输出：中文翻译
        })
    out_json = os.path.join(args.outdir, "cascade_results.json")
    json.dump(results, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    summary = {"asr_model": args.asr_model, "llm": args.llm, "num_samples": len(samples),
               "asr_time_sec": round(asr_time, 1), "llm_time_sec": round(llm_time, 1),
               "total_time_sec": round(asr_time + llm_time, 1)}
    json.dump(summary, open(os.path.join(args.outdir, "c3_summary.json"), "w"),
              ensure_ascii=False, indent=2)

    # ===== 打印样例 =====
    print("\n---- 级联结果样例（语音 → 英文 → 中文）----")
    for r in results[:6]:
        print(f"[{r['id']}]")
        print(f"  ① ASR英文 : {r['asr_text']}")
        print(f"  ② 中文翻译: {r['translation_zh']}")
    print(f"\n已保存中间+最终结果: {out_json}")
    print(f"级联总耗时: {asr_time+llm_time:.1f}s (ASR {asr_time:.1f}s + LLM {llm_time:.1f}s)")


if __name__ == "__main__":
    main()
