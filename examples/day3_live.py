import os
import time
from dotenv import load_dotenv

# Load environment variables FIRST before importing guardrail components
# which might try to read os.environ at import time or initialization time.
load_dotenv()

from guardrail.config import GuardrailConfig
from guardrail.middleware import GuardrailMiddleware

def main():
    print("="*60)
    print("Guardrail Day 3 Live Demo")
    print("="*60)
    print("1. Enter text containing PII or secrets (e.g. 'my email is a@b.com').")
    print("2. Observe the cache in action: identical inputs take ~0ms on second try.")
    print("3. While this script is running, open policy.yaml in another editor.")
    print("4. Change latency_budget_ms, or toggle a category (enabled: false).")
    print("5. Run the same input again. The cache will be invalidated and the new policy applies.")
    print("6. Type 'exit' to quit.")
    print("="*60)
    
    cfg = GuardrailConfig.from_env()
    print(f"Loaded policy from: {cfg.policy_path}")
    
    # Initialize middleware (and PolicyWatcher)
    middleware = GuardrailMiddleware(cfg)
    
    try:
        while True:
            try:
                user_input = input("\n[USER]> ")
                if user_input.strip().lower() in ("exit", "quit"):
                    break
                if not user_input.strip():
                    continue
                
                t0 = time.time()
                resp = middleware.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=100,
                    messages=[{"role": "user", "content": user_input}]
                )
                t1 = time.time()
                
                v = resp.guardrail_verdict
                sr = v.sanitization_result
                meta = sr.sanitization_metadata
                
                print(f"  [Guardrail] Action : {sr.action.value}")
                print(f"  [Guardrail] Latency: {(t1 - t0)*1000:.1f}ms")
                print(f"  [Guardrail] Cache  : {'HIT' if meta.cache_hit else 'MISS'} (Policy v{meta.policy_version})")
                
                if sr.action.value == "REDACT":
                    print(f"  [Guardrail] Cleaned: {sr.sanitized_text}")
                elif sr.action.value == "ALLOW":
                    # For ALLOW, the input was passed to LLM unmodified.
                    # We didn't actually call the LLM to get a reply in this demo to save cost/time,
                    # but normally resp.content would have the LLM's answer.
                    pass
                
            except Exception as e:
                # E.g. GuardrailBlockedError
                print(f"  [Guardrail] Blocked / Error: {e}")
                
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down watcher...")
        middleware.shutdown()


if __name__ == "__main__":
    main()
