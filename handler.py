import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("=" * 50)
print("🚀 Mistral 7B Handler 시작")
print("=" * 50)

# ============================================
# 모델 로드 (컨테이너 시작 시 1번만 실행)
# ============================================

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

print(f"📦 모델 로딩 중: {MODEL_ID}")

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 모델 로드 (4-bit 양자화)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,           # 4-bit 양자화 (메모리 절약)
    device_map="auto",           # GPU 자동 할당
    torch_dtype=torch.float16,   # FP16 사용
    trust_remote_code=True       # 코드 실행 허용
)

print("✅ 모델 로딩 완료!")
print(f"📊 사용 가능한 GPU: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


# ============================================
# Handler 함수 (각 요청마다 실행)
# ============================================

def handler(job):
    """
    RunPod이 API 요청마다 호출하는 함수
    
    Args:
        job (dict): {
            "input": {
                "prompt": str,
                "max_tokens": int (optional),
                "temperature": float (optional)
            }
        }
    
    Returns:
        dict: {
            "output": str,
            "tokens_generated": int
        }
    """
    
    try:
        # 입력 데이터 추출
        job_input = job.get("input", {})
        
        # 필수 파라미터 확인
        if "prompt" not in job_input:
            return {"error": "Missing required field: 'prompt'"}
        
        prompt = job_input["prompt"]
        max_tokens = job_input.get("max_tokens", 200)
        temperature = job_input.get("temperature", 0.7)
        
        print(f"\n{'='*50}")
        print(f"📨 요청 수신:")
        print(f"   Prompt: {prompt[:100]}...")
        print(f"   Max tokens: {max_tokens}")
        print(f"   Temperature: {temperature}")
        print(f"{'='*50}\n")
        
        # Mistral Instruct 형식으로 변환
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        
        # 추론
        print("🤖 추론 시작...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # 결과 디코딩
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # [INST] 부분 제거 (응답만 추출)
        if "[/INST]" in result:
            result = result.split("[/INST]")[-1].strip()
        
        tokens_generated = len(outputs[0])
        
        print(f"✅ 추론 완료!")
        print(f"   생성된 토큰: {tokens_generated}")
        print(f"   응답 길이: {len(result)} 글자\n")
        
        return {
            "output": result,
            "tokens_generated": tokens_generated,
            "model": MODEL_ID
        }
        
    except Exception as e:
        print(f"❌ 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "error": str(e),
            "type": type(e).__name__
        }


# ============================================
# RunPod Serverless 시작
# ============================================

if __name__ == "__main__":
    print("\n🎯 RunPod Serverless 대기 중...")
    print("API 요청을 기다리고 있습니다.\n")
    
    runpod.serverless.start({"handler": handler})