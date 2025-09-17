import requests
import json
from typing import List, Dict, Optional
import time

class SemanticScholarAPI:
    """
    A class to interact with the Semantic Scholar API for research paper search and recommendations.
    """
    
    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {
            'Content-Type': 'application/json'
        }
    
    def search_papers(self, query: str, limit: int = 10, fields: List[str] = None) -> List[Dict]:
        """
        Search for papers using a query string.
        
        Args:
            query (str): Search query
            limit (int): Number of results to return
            fields (List[str]): Fields to include in response
            
        Returns:
            List[Dict]: List of paper objects
        """
        if fields is None:
            fields = [
                'paperId', 'title', 'abstract', 'authors', 'year', 'citationCount',
                'influentialCitationCount', 'venue', 'url', 'publicationDate'
            ]
        
        url = f"{self.base_url}/paper/search"
        params = {
            'query': query,
            'limit': limit,
            'fields': ','.join(fields)
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            return data.get('data', [])
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching papers: {e}")
            return []
    
    def get_paper_details(self, paper_id: str, fields: List[str] = None) -> Optional[Dict]:
        """
        Get detailed information about a specific paper.
        
        Args:
            paper_id (str): Semantic Scholar paper ID
            fields (List[str]): Fields to include in response
            
        Returns:
            Optional[Dict]: Paper details or None if not found
        """
        if fields is None:
            fields = [
                'paperId', 'title', 'abstract', 'authors', 'year', 'citationCount',
                'influentialCitationCount', 'venue', 'url', 'publicationDate',
                'references', 'citations'
            ]
        
        url = f"{self.base_url}/paper/{paper_id}"
        params = {'fields': ','.join(fields)}
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting paper details: {e}")
            return None
    
    def get_recommendations(self, paper_id: str, limit: int = 5) -> List[Dict]:
        """
        Get paper recommendations based on a given paper.
        
        Args:
            paper_id (str): Reference paper ID
            limit (int): Number of recommendations
            
        Returns:
            List[Dict]: List of recommended papers
        """
        url = f"{self.base_url}/paper/{paper_id}/recommendations"
        params = {
            'limit': limit,
            'fields': 'paperId,title,abstract,authors,year,citationCount,venue'
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            return data.get('recommendedPapers', [])
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    def search_by_keywords(self, keywords: List[str], limit: int = 10) -> List[Dict]:
        """
        Search papers by multiple keywords.
        
        Args:
            keywords (List[str]): List of keywords
            limit (int): Number of results
            
        Returns:
            List[Dict]: List of papers
        """
        query = " AND ".join(keywords)
        return self.search_papers(query, limit)
    
    def get_similar_papers(self, paper_title: str, limit: int = 5) -> List[Dict]:
        """
        Find papers similar to a given title.
        
        Args:
            paper_title (str): Title of the reference paper
            limit (int): Number of similar papers
            
        Returns:
            List[Dict]: List of similar papers
        """
        # First search for the paper by title
        search_results = self.search_papers(paper_title, limit=1)
        
        if not search_results:
            return []
        
        paper_id = search_results[0]['paperId']
        return self.get_recommendations(paper_id, limit)
    
    def filter_papers_by_year(self, papers: List[Dict], min_year: int = None, max_year: int = None) -> List[Dict]:
        """
        Filter papers by publication year.
        
        Args:
            papers (List[Dict]): List of papers
            min_year (int): Minimum year
            max_year (int): Maximum year
            
        Returns:
            List[Dict]: Filtered papers
        """
        filtered = []
        for paper in papers:
            year = paper.get('year')
            if year:
                if min_year and year < min_year:
                    continue
                if max_year and year > max_year:
                    continue
                filtered.append(paper)
        return filtered
    
    def sort_papers_by_citations(self, papers: List[Dict], descending: bool = True) -> List[Dict]:
        """
        Sort papers by citation count.
        
        Args:
            papers (List[Dict]): List of papers
            descending (bool): Sort in descending order
            
        Returns:
            List[Dict]: Sorted papers
        """
        return sorted(
            papers,
            key=lambda x: x.get('citationCount', 0),
            reverse=descending
        )
    
    def format_paper_info(self, paper: Dict) -> str:
        """
        Format paper information for display.
        
        Args:
            paper (Dict): Paper object
            
        Returns:
            str: Formatted paper information
        """
        title = paper.get('title', 'Unknown Title')
        authors = ', '.join([author.get('name', '') for author in paper.get('authors', [])])
        year = paper.get('year', 'Unknown Year')
        citations = paper.get('citationCount', 0)
        venue = paper.get('venue', 'Unknown Venue')
        abstract = paper.get('abstract', 'No abstract available')
        
        return f"""
**Title:** {title}
**Authors:** {authors}
**Year:** {year}
**Citations:** {citations}
**Venue:** {venue}
**Abstract:** {abstract[:300]}...
        """.strip()