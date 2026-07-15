# -*- coding: utf-8 -*-
"""
C3 级联语音翻译系统 —— 原生关闭思考链无报错版
范式：ASR (Whisper-small) → LLM 级联，英文语音→中文文本
支持Qwen2.5 / Qwen3，原生禁用推理块，无分割字符串报错
"""

import os, sys, json, time, argparse

# 国内镜像加速
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 离线模式自动识别逻辑
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
    ap.add_argument("--offline", action="store_true", help="离线模式：不联网，读取本地模型")
    ap.add_argument("--outdir", default=os.path.join(HERE, "..", "outputs"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 加载数据集
    ds = json.load(open(args.dataset, encoding="utf-8"))
    data_root = os.path.dirname(os.path.abspath(args.dataset))
    samples = ds[args.start:] if args.n <= 0 else ds[args.start:args.start + args.n]
    audio_paths = [os.path.join(data_root, s["audio"]) for s in samples]
    print(f"数据集: {os.path.abspath(args.dataset)}  （共 {len(ds)} 条，本次使用 {len(samples)} 条）")
    device = 0 if torch.cuda.is_available() else -1

    # ===================== 第一级 ASR 语音转英文 =====================
    print("=" * 60)
    print("第一级 ASR：Whisper-small  语音 → 英文文本")
    print("=" * 60)
    t0_asr = time.time()
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.asr_model,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
        device=device
    )
    asr_out = asr(
        audio_paths,
        batch_size=16,
        generate_kwargs={"language": "english", "task": "transcribe"}
    )
    asr_texts = [o["text"].strip() for o in asr_out]
    asr_time = time.time() - t0_asr
    print(f"ASR 完成，{len(samples)} 条，耗时 {asr_time:.1f}s")

    # ===================== 第二级 LLM 英文转中文 =====================
    print("\n" + "=" * 60)
    llm_name = args.llm.split("/")[-1]
    print(f"第二级 LLM：{llm_name}  英文文本 → 中文翻译")
    print("=" * 60)
    t0_llm = time.time()
    # 离线加载tokenizer与模型
    tok = AutoTokenizer.from_pretrained(args.llm, local_files_only=args.offline)
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
        local_files_only=args.offline,
        device_map="auto"
    )
    translations = []
    for eng_txt in asr_texts:
        # 翻译提示词
        system_prompt = """你是专业英文翻译，只输出通顺中文，禁止任何解释、英文原文、多余符号、换行"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": eng_txt}
        ]
        # 关键：enable_thinking=False 官方原生关闭思考链，不会输出
        full_prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tok(full_prompt, return_tensors="pt", padding=True).to(llm.device)
        with torch.no_grad():
            # Qwen3不支持temperature，删除该参数
            outputs = llm.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        # 截取生成部分文本
        input_token_len = inputs["input_ids"].shape[1]
        raw_output = tok.decode(outputs[0][input_token_len:], skip_special_tokens=True)
        # 仅简单清理换行空格，无需处理
        clean_output = raw_output.replace("\n", "").replace("\r", "").strip()
        translations.append(clean_output)
    llm_time = time.time() - t0_llm
    print(f"LLM 翻译完成，{len(samples)} 条，耗时 {llm_time:.1f}s")

    # ===================== 保存级联结果 =====================
    result_list = []
    for sample, asr_en, zh_tr in zip(samples, asr_texts, translations):
        result_list.append({
            "id": sample["id"],
            "audio": sample["audio"],
            "reference_en": sample["text"],
            "asr_text": asr_en,
            "translation_zh": zh_tr
        })
    out_json_path = os.path.join(args.outdir, "cascade_results_ex.json")
    json.dump(result_list, open(out_json_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 保存耗时汇总文件（任务5延迟分析用）
    summary_info = {
        "asr_model": args.asr_model,
        "llm": args.llm,
        "num_samples": len(samples),
        "asr_time_sec": round(asr_time, 1),
        "llm_time_sec": round(llm_time, 1),
        "total_time_sec": round(asr_time + llm_time, 1)
    }
    summary_path = os.path.join(args.outdir, "c3_summary_ex.json")
    json.dump(summary_info, open(summary_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 打印前6条样例
    print("\n---- 级联结果样例（语音 → 英文 → 中文）----")
    for res in result_list[:6]:
        print(f"[{res['id']}]")
        print(f"  ① ASR英文 : {res['asr_text']}")
        print(f"  ② 中文翻译: {res['translation_zh']}")
    print(f"\n已保存中间+最终结果: {out_json_path}")
    print(f"级联总耗时: {asr_time+llm_time:.1f}s (ASR {asr_time:.1f}s + LLM {llm_time:.1f}s)")

if __name__ == "__main__":
    main()