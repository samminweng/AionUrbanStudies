// Create a topic list view
function ClusterTopicListView(cluster_data, cluster_docs, accordion_div){

    // Populate the topic list with given length
    function populateTopicList(cluster_topics, max_length, p_div) {
        const p = p_div.find('p');
        p.empty();
        for (let i = 0; i < max_length; i++) {
            const topic = cluster_topics[i];
            const link = $('<button type="button" class="btn btn-link">'
                + topic['topic'] + ' (' + topic['doc_ids'].length + ') </button>');
            link.button();
            link.click(function (event) {
                $('#topics').val(topic['topic']);
                // // Get the documents about the topic
                const topic_docs = cluster_docs.filter(d => topic['doc_ids'].includes(parseInt(d['DocId'])));
                const doc_list = new DocList(cluster_data, topic_docs, topic);
            });
            p.append(link);
        }
    }

    // Create a list of topics
    function createTopicParagraphs(cluster_topics) {
        const p_div = $('<div class="topic_list"><p></p></div>');
        populateTopicList(cluster_topics, 30, p_div);   // Display top 30 topics
        // Add the view more button
        const view_more_btn = $('<button>View More <span class="ui-icon ui-icon-plus"></button>');
        const view_less_btn = $('<button>View Less <span class="ui-icon ui-icon-minus"></button>');
        view_more_btn.button();
        view_less_btn.button();
        // By default, display 'view more' and hide 'view less'
        view_less_btn.hide();
        p_div.append(view_more_btn);
        p_div.append(view_less_btn);
        // Define view more btn click event
        view_more_btn.click(function (event) {
            // Display top 100 topics
            populateTopicList(cluster_topics, 100, p_div);
            // Hide view more
            view_more_btn.hide();
            view_less_btn.show();
        });
        // Define view less btn click event
        view_less_btn.click(function (event) {
            // Display top 30 topics
            populateTopicList(cluster_topics, 30, p_div);
            // Hide view more
            view_more_btn.show();
            view_less_btn.hide();
        });
        return p_div;
    }

    function _createUI(){
        // Display the topics by keyword extraction
        const cluster_topics = cluster_data['Topics'];
        accordion_div.append($('<h3><span class="fw-bold"> Top 30 Topics </span></h3>'));
        const topic_div = $('<div class="topics"></div>');
        const sort_widget = $('<div>' +
            '<span>Sort by </span>' +
            '<div class="form-check form-check-inline">' +
            '   <label class="form-check-label" for="score">Score</label>' +
            '   <input class="form-check-input" type="radio" name="sort-btn" value="score" checked>' +
            '</div>' +
            '<div class="form-check form-check-inline">' +
            '   <label class="form-check-label" for="count">Count</label>' +
            '   <input class="form-check-input" type="radio" name="sort-btn" value="count" >' +
            '</div>' +
            '</div>');
        // Sort the topics by count (default)
        cluster_topics.sort((a, b) => b['score'] - a['score']);
        console.log(cluster_topics);
        // Add a radio button to sort topics by scores or range
        topic_div.append(sort_widget);
        const topic_p = createTopicParagraphs(cluster_topics);
        topic_div.append(topic_p); // Show topics
        // // Add the on click event to radio button
        sort_widget.find('input[name=sort-btn]').change(function () {
            const sorted_index = sort_widget.find('input[name=sort-btn]:checked').val();
            const sorted_topic = cluster_data['Topics'];
            if (sorted_index === 'count') {
                // Sort the topic by count
                sorted_topic.sort((a, b) => b['doc_ids'].length - a['doc_ids'].length);
            } else {
                sorted_topic.sort((a, b) => b['score'] - a['score']);// Sort by score
            }
            // console.log(topics);
            topic_div.find(".topic_list").remove();
            topic_div.append(createTopicParagraphs(sorted_topic));
        });
        accordion_div.append(topic_div);
    }

    _createUI();

}
