def encrypt_vigenere(plaintext: str, keyword: str) -> str:
    """
    Encrypts plaintext using a Vigenere cipher.

    >>> encrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> encrypt_vigenere("python", "a")
    'python'
    >>> encrypt_vigenere("ATTACKATDAWN", "LEMON")
    'LXFOPVEFRNHR'
    """

    lowercase_start = ord('a')
    uppercase_start = ord('A')
    alphabet_size = 26
    key_len = len(keyword)

    ciphertext = ""

    for i in range(len(plaintext)):
        char = plaintext[i]

        key_letter_ord = ord(keyword[i % key_len])
        if (key_letter_ord >= lowercase_start) and (key_letter_ord < lowercase_start + alphabet_size):
            shift = key_letter_ord - lowercase_start
        elif (key_letter_ord >= uppercase_start) and (key_letter_ord < uppercase_start + alphabet_size):
            shift = key_letter_ord - uppercase_start
        else:
            shift = 0

        if (ord(char) >= lowercase_start) and (ord(char) < lowercase_start + alphabet_size):
            new_char = chr((ord(char) - lowercase_start + shift) % alphabet_size + lowercase_start)
            ciphertext += new_char
        elif (ord(char) >= uppercase_start) and (ord(char) < uppercase_start + alphabet_size):
            new_char = chr((ord(char) - uppercase_start + shift) % alphabet_size + uppercase_start)
            ciphertext += new_char
        else:
            ciphertext += char

    return ciphertext



    # PUT YOUR CODE HERE
    return ciphertext


def decrypt_vigenere(ciphertext: str, keyword: str) -> str:
    """
    Decrypts a ciphertext using a Vigenere cipher.

    >>> decrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> decrypt_vigenere("python", "a")
    'python'
    >>> decrypt_vigenere("LXFOPVEFRNHR", "LEMON")
    'ATTACKATDAWN'
    """
    lowercase_start = ord('a')
    uppercase_start = ord('A')
    alphabet_size = 26
    key_len = len(keyword)

    plaintext = ""

    for i in range(len(ciphertext)):
        char = ciphertext[i]

        key_letter_ord = ord(keyword[i % key_len])
        if (key_letter_ord >= lowercase_start) and (key_letter_ord < lowercase_start + alphabet_size):
            shift = key_letter_ord - lowercase_start
        elif (key_letter_ord >= uppercase_start) and (key_letter_ord < uppercase_start + alphabet_size):
            shift = key_letter_ord - uppercase_start
        else:
            shift = 0

        if (ord(char) >= lowercase_start) and (ord(char) < lowercase_start + alphabet_size):
            new_char = chr((ord(char) - lowercase_start - shift) % alphabet_size + lowercase_start)
            plaintext += new_char
        elif (ord(char) >= uppercase_start) and (ord(char) < uppercase_start + alphabet_size):
            new_char = chr((ord(char) - uppercase_start - shift) % alphabet_size + uppercase_start)
            plaintext += new_char
        else:
            plaintext += char

    return plaintext
