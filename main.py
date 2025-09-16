from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import argparse


class TextGenerator:
    """
    генерация текста с использованием модели ruGPT-3.5
    """
    def __init__(self, model_name: str = "ai-forever/ruGPT-3.5-13B"):
        """
        иниц модели и токенизатора
        :param model_name: название модели из HuggingFace Hub
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_4bit=True
            )

            self.model = self.model.to(
                "cuda:0" if torch.cuda.is_available() else "cpu"
                )
            
        except Exception as e:
            raise RuntimeError(f"ошибка при загрузке модели {model_name}: {e}")

    def predict(
        self,
        request: str,
        temperature: float = 0.8,
        max_new_tokens: int = 100,
        repetition_penalty: float = 1.2,
        num_beams: int = 2,
        no_repeat_ngram_size: int = 3
    ) -> str:
        """
        генерация текста на основе строки request
        :param request: текст запроса
        :param temperature: креативность генерации
        :param max_new_tokens: максимальное число новых токенов
        :param repetition_penalty: штраф за повторение токенов
        :param num_beams: количество лучей в beam search
        :param no_repeat_ngram_size: запрет повторов n-грамм
        :return: сгенерированный текст
        """
        if not isinstance(request, str) or not request.strip():
            raise ValueError("request должен быть непустой строкой")

        try:
            encoded_input = self.tokenizer(
                request,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.model.device)

            output = self.model.generate(
                **encoded_input,
                num_beams=num_beams,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=self.tokenizer.eos_token_id
            )

            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"ошибка генерации текста: {e}")


def parse_args():
    """
    парсер аргументов командной строки
    """
    parser = argparse.ArgumentParser(description="генерация текста с ruGPT-3.5")

    parser.add_argument(
        "--request", "-r", type=str, required=True,
        help="текстовый запрос (prompt)"
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.8,
        help="температура (креативность) генерации (по умолчанию 0.8)"
    )
    parser.add_argument(
        "--max_tokens", "-m", type=int, default=100,
        help="максимальное количество новых токенов (по умолчанию 100)"
    )
    parser.add_argument(
        "--repetition_penalty", "-p", type=float, default=1.2,
        help="штраф за повторение токенов (по умолчанию 1.2)"
    )
    parser.add_argument(
        "--num_beams", "-b", type=int, default=2,
        help="кол-во beam search путей (по умолчанию 2)"
    )
    parser.add_argument(
        "--no_repeat_ngram_size", "-n", type=int, default=3,
        help="запрет на повторы n-грамм данного размера (по умолчанию 3)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        generator = TextGenerator()

        result = generator.predict(
            request=args.request,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            no_repeat_ngram_size=args.no_repeat_ngram_size
        )

    except Exception as err:
        print(f"{err}")
