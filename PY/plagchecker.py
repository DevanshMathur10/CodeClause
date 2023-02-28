# Simhash is a technique used for detecting near-duplicate content in large datasets, such as documents or web pages.
#  It works by generating a "fingerprint" or hash of a piece of content, which is then compared to the hashes
#  of other pieces of content to identify similarities.

# The Simhash algorithm uses a technique called "shingling" to create a set of features from the input text.
#  These features are then hashed using a cryptographic hash function, such as MD5 or SHA-1,
#  to produce a fixed-length hash code. Similar documents will have similar hashes, which can be identified using
#  a "hamming distance" metric to compare the similarity of two hashes.

# Simhash is particularly useful for detecting near-duplicate content in large datasets, 
# as it is computationally efficient and can be scaled to handle large volumes of data. 
# It is commonly used in applications such as plagiarism detection, spam filtering, and clustering similar documents.

# One limitation of Simhash is that it may not be effective at detecting content that has been
#  intentionally modified to evade detection, such as by using synonyms or paraphrasing. 
# In these cases, more advanced techniques, such as natural language processing, 
# may be required to accurately detect plagiarism or other forms of duplicate content.

import simhash

# Function that takes two texts as input and returns their similarity score
def plagiarism_checker(text1, text2):
    # Generate Simhash values for the two texts
    hash1 = simhash.Simhash(text1)
    hash2 = simhash.Simhash(text2)
    
    # Calculate the "distance" between the two hashes and convert to a similarity score
    similarity = 1 - hash1.distance(hash2)
    
    # Return the similarity score
    return similarity

# Initialize two empty strings to store the contents of the input files
text1 = ""
text2 = ""

# Read the contents of the first input file into text1
with open("INTERNSHIPS/CODECLAUSE/PY/text1.txt",'r') as file:
    for line in file:
        text1 += line.strip()

# Read the contents of the second input file into text2
with open("INTERNSHIPS/CODECLAUSE/PY/text2.txt",'r') as file:
    for line in file:
        text2 += line.strip()

# Call the plagiarism_checker function with the two input texts and print the result
print("Similarity:", plagiarism_checker(text1, text2))
