def first_n_words(string, n):
    words = string.split()  # Split the string into words
    return ' '.join(words[:n])

string = "This is"
n = 3
result = first_n_words(string, n)
print(result)