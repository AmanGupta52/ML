# importing libraries
from bs4 import BeautifulSoup
import requests

def main(URL):

    # opening our output file in append mode
    File = open("out.csv", "a", encoding="utf-8")

    # specifying user agent
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.5'
    }

    # Making the HTTP request
    webpage = requests.get(URL.strip(), headers=HEADERS)

    # Creating the Soup Object
    soup = BeautifulSoup(webpage.content, "lxml")

    # -------------------- PRODUCT TITLE --------------------
    try:
        title = soup.find("span", attrs={"id": "productTitle"})
        title_string = title.get_text(strip=True).replace(",", "")
    except:
        title_string = "NA"

    print("Product Title =", title_string)
    File.write(f"{title_string},")

    # -------------------- PRICE --------------------
    try:
        price = soup.find("span", attrs={"class": "a-price-whole"})
        price_string = price.get_text(strip=True).replace(",", "")
    except:
        price_string = "NA"

    print("Product Price =", price_string)
    File.write(f"{price_string},")

    # -------------------- RATING --------------------
    try:
        rating = soup.find("span", attrs={"class": "a-icon-alt"})
        rating_string = rating.get_text(strip=True).replace(",", "")
    except:
        rating_string = "NA"

    print("Overall Rating =", rating_string)
    File.write(f"{rating_string},")

    # -------------------- REVIEW COUNT --------------------
    try:
        review_count = soup.find("span", attrs={"id": "acrCustomerReviewText"})
        review_string = review_count.get_text(strip=True).replace(",", "")
    except:
        review_string = "NA"

    print("Total Reviews =", review_string)
    File.write(f"{review_string},")

    # -------------------- AVAILABILITY --------------------
    try:
        available = soup.find("div", attrs={"id": "availability"})
        available_string = available.find("span").get_text(strip=True).replace(",", "")
    except:
        available_string = "NA"

    print("Availability =", available_string)
    File.write(f"{available_string}\n")

    File.close()



# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    file = open("url.txt", "r")

    # iterate over URLs
    for link in file.readlines():
        print("\nScraping:", link.strip())
        main(link)

    file.close()
