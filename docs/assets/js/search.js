document.addEventListener('DOMContentLoaded', () => {
  const searchInput = document.getElementById('search-input');
  const searchResults = document.getElementById('search-results');
  let searchData = [];

  if (searchInput && searchResults) {
    const baseUrl = window.SITE_BASE_URL || '';
    
    // Fetch search data
    fetch(baseUrl + '/search.json')
      .then(response => response.json())
      .then(data => {
        searchData = data;
      })
      .catch(error => console.error('Error fetching search data:', error));

    searchInput.addEventListener('input', (e) => {
      const query = e.target.value.toLowerCase();
      searchResults.innerHTML = '';
      
      if (query.length === 0) {
        searchResults.style.display = 'none';
        return;
      }

      const results = searchData.filter(post => 
        post.title.toLowerCase().includes(query) || 
        post.excerpt.toLowerCase().includes(query)
      );

      if (results.length > 0) {
        results.forEach(post => {
          const resultItem = document.createElement('a');
          resultItem.href = post.url;
          resultItem.className = 'search-result-item';
          resultItem.innerHTML = `
            <div class="search-result-title">${post.title}</div>
            <div class="search-result-excerpt">${post.excerpt}</div>
          `;
          searchResults.appendChild(resultItem);
        });
        searchResults.style.display = 'block';
      } else {
        searchResults.innerHTML = '<div class="search-result-item">No results found.</div>';
        searchResults.style.display = 'block';
      }
    });

    // Hide results when clicking outside
    document.addEventListener('click', (e) => {
      if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
        searchResults.style.display = 'none';
      }
    });
    
    // Show results when clicking input if there's a query
    searchInput.addEventListener('focus', () => {
      if (searchInput.value.length > 0 && searchResults.innerHTML !== '') {
        searchResults.style.display = 'block';
      }
    });
  }
});
