def decode(preds, idx_to_char):
    decoded = []

    for seq in preds:
        prev = None
        text = ""

        for token in seq:
            token = token.item()

            # 0 là padding
            if token == 0:
                continue

            char = idx_to_char[token]

            # 🔥 BỎ dấu gạch ngang
            if char == "-":
                prev = char
                continue

            # Loại bỏ ký tự lặp
            if char != prev:
                text += char

            prev = char

        decoded.append(text)

    return decoded