"""
jinli yu
chromedriver is required
e.g. python crawler.py harry\ potter
"""
from urllib.parse import quote
from bs4 import BeautifulSoup
from selenium import webdriver
import sys

def get_review_html(movie_name):
    try:
        search_url = "https://www.rottentomatoes.com/search/?search="+quote(movie_name) #url to search by movie name
        option = webdriver.ChromeOptions()
        option.add_argument('headless')
        driver = webdriver.Chrome(chrome_options=option)
        driver.get(search_url)
        driver.find_element_by_xpath("//section[@id='movieSection']//a[1]").click() #first movie in the list
        driver.find_element_by_xpath("//section[@id='audience_reviews']//div[@class='view-all']/a").click()#audiance reviews
        html = driver.page_source
    except Exception as e:
        print("Cannot find this movie or there is no audience reviews for this movie now!")
        html = ""
    finally:
        driver.close()
        return html

def get_reviews(html):
    soup = BeautifulSoup(html,"html.parser")
    reviews_tag = soup.select(".review_table_row")
    reviews = [{
            "reviewer" : review_tag.span.string,
            "date" : review_tag.contents[3].contents[3].string,
            "score" : review_tag.select(".scoreWrapper")[0].span.attrs['class'][0],
            "comment" : review_tag.select(".user_review")[0].text
        }for review_tag in reviews_tag]
    return reviews

if __name__ == "__main__":
    movie_name = sys.argv[1]
    html = get_review_html(movie_name)
    if(len(html)==0):
        sys.exit()
    reviews = get_reviews(html)
    #write reviews to file
    fw = open("./reviews for "+movie_name,"w")
    for review in reviews:
        fw.write(str(review))
    fw.close()
