from app import config
from app.embeddings import encode_passages, encode_query


def main() -> None:
    passages = ["Доставка замовлення триває два робочі дні.",
                "Навушники заряджаються через USB-C."]
    vectors = encode_passages(passages)
    query = encode_query("Скільки чекати на посилку?")
    print(f"Модель: {config.EMBEDDING_MODEL}")
    print(f"Розмір вектора: {vectors.shape[1]}")
    print(f"Контрольна схожість: {float(vectors[0] @ query):.4f}")


if __name__ == "__main__":
    main()
