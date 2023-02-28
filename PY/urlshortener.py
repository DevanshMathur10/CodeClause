import hashlib

# A dictionary to store the mappings from short URLs to long URLs
url_map = {}

def shorten_url(url):
    # Generate a unique hash for the URL
    hash = hashlib.sha1(url.encode('utf-8')).hexdigest()[:6]

    # Keep generating new hashes until a unique one is found
    while hash in url_map.keys():
        hash = hashlib.sha1((hash + url).encode('utf-8')).hexdigest()[:6]

    # Store the mapping from the short URL to the long URL
    url_map[hash] = url

    # Create the shortened URL by appending the hash to a base URL
    shortened_url = "https://killswitch.ly/" + hash

    return shortened_url

# Test the shortener
long_url = "https://www.linkedin.com/feed/update/urn:li:activity:7016669137749864449/"
short_url = shorten_url(long_url)
print("Short URL:", short_url)
