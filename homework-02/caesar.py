import typing as tp


def encrypt_caesar(plaintext: str, shift: int = 3) -> str:
    """
    Encrypts plaintext using a Caesar cipher.

    >>> encrypt_caesar("PYTHON")
    'SBWKRQ'
    >>> encrypt_caesar("python")
    'sbwkrq'
    >>> encrypt_caesar("Python3.6")
    'Sbwkrq3.6'
    >>> encrypt_caesar("")
    ''
    """

    lowercase_start = ord('a')
    uppercase_start = ord('A')
    alphabet_size = 26

    ciphertext = ""

    for char in plaintext:
        if (ord(char) >= lowercase_start) and (ord(char) < lowercase_start + alphabet_size):
            new_char = chr((ord(char) - lowercase_start + shift) % alphabet_size + lowercase_start)
            ciphertext += new_char
        elif (ord(char) >= uppercase_start) and (ord(char) < uppercase_start + alphabet_size):
            new_char = chr((ord(char) - uppercase_start + shift) % alphabet_size + uppercase_start)
            ciphertext += new_char
        else:
            ciphertext += char

    return ciphertext


def decrypt_caesar(ciphertext: str, shift: int = 3) -> str:
    """
    Decrypts a ciphertext using a Caesar cipher.

    >>> decrypt_caesar("SBWKRQ")
    'PYTHON'
    >>> decrypt_caesar("sbwkrq")
    'python'
    >>> decrypt_caesar("Sbwkrq3.6")
    'Python3.6'
    >>> decrypt_caesar("")
    ''
    """
    plaintext = encrypt_caesar(ciphertext, -shift)
    return plaintext


def caesar_breaker_brute_force(ciphertext: str, dictionary: tp.Set[str]) -> int:
    """
    Brute force breaking a Caesar cipher.
    """
    best_shift = 0

    word_list = ciphertext.split()
    first_meaning_word = next((word for word in word_list if len(word) > 3), word_list[0])

    for shift in range(26):
        plaintext = decrypt_caesar(first_meaning_word, shift)
        if plaintext in dictionary:
            best_shift = shift
            break

    return best_shift
