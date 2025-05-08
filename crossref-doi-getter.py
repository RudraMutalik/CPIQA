import json
import os
import time
import requests


core_api_key = os.getenv('CORE_API_KEY')


def exponential_backoff_request(method, url, retries=5, backoff_factor=1, **kwargs):
    """
    Make a request with exponential backoff.

    Args:
        method (str): HTTP method (e.g., 'get', 'post').
        url (str): The URL for the request.
        retries (int): Number of retries before giving up.
        backoff_factor (float): Factor to calculate the backoff delay.
        kwargs: Additional arguments for the requests method.

    Returns:
        Response: The response object if the request is successful.

    Raises:
        requests.exceptions.RequestException: If all retries fail.
    """
    for attempt in range(retries):
        try:
            response = getattr(requests, method)(url, **kwargs)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                wait_time = backoff_factor * (2 ** attempt)  # Exponential backoff
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Raising exception.")
                raise

def fetch_article_by_doi(doi, api_key):
    base_url = "https://api.core.ac.uk/v3/discover"
    headers = {
        #"Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    query = {
        "doi": f"{doi}"
    }
    try:
        #response = requests.post(base_url, json=query, headers=headers, timeout=10)
        response = exponential_backoff_request('post', base_url, json=query, headers=headers, timeout=10)
        #response.raise_for_status()
        data = response.json()

        if data.get("fullTextLink"):
            return data.get("fullTextLink")
        else:
            print(f"No articles found for the provided DOI: {doi}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from CORE API: {e}")
        return None


def fetch_dois(issn, from_year=2024):
    base_url = "https://api.crossref.org/works"
    #dois = []

    metadata = []

    try:
        response = exponential_backoff_request('get', base_url,
                                               params={
                                                   "filter": f"issn:{issn},from-pub-date:{from_year},until-pub-date:{from_year}",
                                                   "rows": 1000},
                                               )
        #response = requests.get(
        #    base_url,
        #    params={"filter": f"issn:{issn},from-pub-date:{from_year},until-pub-date:{from_year}",
        #            "rows": 1000},
        #    #timeout=10  # Set a timeout to avoid hanging
        #)
        response.raise_for_status()  # Handle HTTP errors
        data = response.json()

        # Process items
        items = data["message"].get("items", [])
        #dois.extend(item.get("DOI") for item in items)

        for item in items:
            try:
                meta = {'doi': item.get("DOI"),
                        'title': item.get("title")[0],
                        'author': item.get("author")}
                metadata.append(meta)
            except:
                print(item)

        #print([item.get("title") for item in items])
        #print(items[0].keys())

        # Respect rate limits
        time.sleep(1)

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    return metadata

def download_pdf(doi, url):
    file_name = doi.replace('/', '_')
    file_Path = f'cpiqa/pdfs/{file_name}.pdf'
    if os.path.isfile(file_Path):
        print(f'File already exists: {doi}')
        return

    try:
        response = requests.get(url)
    except:
        print(f'Failed to download file: {doi}')
        return
    #response = exponential_backoff_request('get', url)

    if response.status_code == 200:
        with open(file_Path, 'wb') as file:
            file.write(response.content)
        print(f'File downloaded successfully: {doi}')
    else:
        print(f'Failed to download file: {doi}')

issns = [
    "1432-0894",  # Climate Dynamics
    "1573-1480",  # Climatic Change
    "1097-0088",  # International Journal of Climatology
    "1520-0442",  # Journal of Climate
    "1758-6798",  # Nature Climate Change
    "1752-0908",  # Nature Geoscience
    "1757-7799",  # WIRES Climate Change
    "2364-3587",  # Advances in Statistical Climatology, Meteorology and Oceanography
    "1814-9332",  # Climate of the Past
    "2190-4987",  # Earth System Dynamics
    "1866-3516",  # Earth System Science Data
    "2569-7110",  # Geoscience Communication
]

year = 2020


with open('dois.txt', 'r') as f:
    done = [line for line in f]

papers = []

if True:
    for issn in issns:
        all_papers = fetch_dois(issn, year)
        papers = papers + all_papers
        print(f"Found {len(all_papers)} DOIs in " + str(issn))

    print(f"Total DOIs: {len(papers)}")

    with open('cpiqa/metadata' + str(year) + '.json', 'w+') as outfile:
        for ddict in papers:
            jout = json.dumps(ddict) + '\n'
            outfile.write(jout)

with open('cpiqa/metadata' + str(year) + '.json', 'r') as infile:
    papers = []
    for line in infile:
        #print(line)
        papers.append(json.loads(line))

for paper in papers:

    if paper["doi"] in done:
        continue

    article_link = fetch_article_by_doi(paper["doi"], core_api_key)

    if article_link:
        #print(f"DOI: {doi} @ {article_link}")
        download_pdf(paper["doi"], article_link)

    with open('dois.txt', 'a') as f:
        f.write('%s\n' % paper["doi"])

