import requests
import os

from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

def scraped_images(url, path):
    def is_valid(url):
        """
        Checks whether `url` is a valid URL.
        """
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)
    
    def get_all_images(url):
        """
        Returns all image URLs on a single `url`
        """
        soup = bs(requests.get(url).content, "html.parser")
        urls = []
        for img in tqdm(soup.find_all("img"), "Extracting images"):
            img_url = img.attrs.get("src")
            if not img_url:
                # if img does not contain src attribute, just skip
                continue
            img_url = urljoin(url, img_url)
            try:
                pos = img_url.index("?")
                img_url = img_url[:pos]
            except ValueError:
                pass
            if is_valid(img_url):
                urls.append(img_url)
        return urls
    
    def download(url, pathname):
        """
        Downloads a file given an URL and puts it in the folder `pathname`
        """
        # if path doesn't exist, make that path dir
        if not os.path.isdir(pathname):
            os.makedirs(pathname)
        # download the body of response by chunk, not immediately
        response = requests.get(url, stream=True)
        # get the total file size
        file_size = int(response.headers.get("Content-Length", 0))
        # get the file name
        filename = os.path.join(pathname, url.split("/")[-1])
        # try:
        #     filename = filename.replace("")
        # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
        progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
        if 'flp' in filename.lower():
            filename = filename.replace('_max_296x197', '')
        with open(filename, "wb") as f:
            if 'logo' not in filename:
                for data in progress.iterable:
                    # write data read to the file
                    f.write(data)
                    # update the progress bar manually
                    progress.update(len(data))
    
    # url = 'https://www.rightmove.co.uk/properties/112099649#/?channel=RES_BUY'
    # path = '/home/fx/Dev/AI_chatbot/Jupyter_Notebook_Test/images'
    imgs = get_all_images(url)
    for img in imgs:
        # for each image, download it
        download(img, path)



def get_energy_data(postcode, number):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get("https://www.gov.uk/find-energy-certificate")
    driver.find_element(By.XPATH, '//*[@id="get-started"]/a').click()

    driver.find_element(By.XPATH, '//*[@id="domestic"]').click()
    driver.find_element(By.XPATH, '//*[@id="domestic"]').click()
    driver.find_element(By.XPATH, '//*[@id="main-content"]/form/fieldset/button').click()

    driver.find_element(By.ID, 'postcode').send_keys(postcode)
    driver.find_element(By.XPATH, '//*[@id="main-content"]/div/div/form/fieldset/button').click()

    links = driver.find_elements(By.CLASS_NAME, 'govuk-link')
    for iter in links:
        if number in iter.text:
            iter.click()
            break
    url = driver.current_url
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    energy_scrape_data = ' '.join(soup.stripped_strings)
    return energy_scrape_data

def get_tax_data(postcode, number):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get("https://www.tax.service.gov.uk/check-council-tax-band/search?_ga=2.95280031.223622977.1663838161-1757065444.1663255247")
    driver.find_element(By.ID, 'postcode').send_keys(postcode)
    driver.find_element(By.CLASS_NAME, 'govuk-button').click()
    
    true_property = None
    while true_property is None:
        properties = driver.find_elements(By.CLASS_NAME, 'govuk-table__row')
        for iter in properties:
            if number in iter.text:
                true_property = iter.text
                iter.click()
                break
        if true_property is None:
            links = driver.find_elements(By.CLASS_NAME, 'voa-pagination__link')
            for iter in links:
                if 'next' in iter.text.lower():
                    driver.get(iter.get_attribute('href'))

    url = driver.current_url
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    scrape_data = ' '.join(soup.stripped_strings)
    return scrape_data            
    


  

    

    