// Create a div to display the grouped key phrases
function KeywordClusterView(keyword_cluster, docs) {
    const group_no = keyword_cluster['Group'];
    const score = keyword_cluster['score'].toFixed(2);
    const keywords = keyword_cluster['Key-phrases'];
    // D3 category color pallets
    const color_plates = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
        "#7f7f7f", "#bcbd22", "#17becf"];
    const color_no = group_no -1;
    const color = color_plates[color_no];

    // Display keywords in accordion
    function displayKeywordList() {
        const keyword_div = $('<div class="container-sm mb-3"></div>');
        // Get the number of rows by rounding upto the closest integer
        const num_row = Math.ceil(keywords.length / 3) + 1;
        // Add each key phrase
        for (let i = 0; i < num_row; i++) {
            // Create a new row
            const row = $('<div class="row"></div>');
            for (let j = 0; j < 3; j++) {
                const index = i + j * num_row;
                const col = $('<div class="col-sm border-bottom text-sm-start text-truncate"></div>')
                if (index < keywords.length) {
                    const keyword = keywords[index];
                    const btn = $('<button type="button" class="btn btn-link btn-sm">' + keyword+ '</button>');
                    // Click the keyword button to display the relevant article
                    btn.button();
                    btn.click(function(event){
                       const matched_docs = docs.filter(d => {
                           const article_keywords = d['KeyPhrases'];
                           const found = article_keywords.find(k => k.toLowerCase() === keyword.toLowerCase());
                           if(found)
                                return true;
                           return false;
                       });
                       const doc_list = new DocList(matched_docs, [keyword], group_no-1);
                    });
                    col.append(btn);
                }
                row.append(col);
            }
            keyword_div.append(row);
        }
        return keyword_div;
    }


    function _createUI() {
        $('#keyword_cluster_view').empty();
        // Heading
        const container = $('<div class="m-3"></div>');
        const heading = $('<div class="mb-3">' +
                          '<span class="fw-bold" style="color:' + color + '">Keyword Cluster ' + group_no + ' </span>' +
                          ' (' + score + ') contains ' + keywords.length + ' keywords' + ' across ' + docs.length + ' articles</div>');
        // A list of grouped key phrases
        container.append(heading);
        container.append(displayKeywordList());
        $('#keyword_cluster_view').append(container);
        const doc_list = new DocList(docs, keywords, color_no);
    }

    _createUI();
}
