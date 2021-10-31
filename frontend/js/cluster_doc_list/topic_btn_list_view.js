// Create an auto-complete
function TopicBtnListView(cluster_no, cluster_topic_words, doc_key_terms){
    // Fill out the topic using TF-IDF
    const cluster_topics = cluster_topic_words.find(c => c['Cluster'] === cluster_no);
    const available_topics = cluster_topics['Topic1-gram'].concat(cluster_topics['Topic2-gram'], cluster_topics['Topic3-gram']);
    // Get the articles in cluster #no
    const cluster_doc_ids = cluster_topics['DocIds'];
    const cluster_docs = doc_key_terms.filter(d => cluster_doc_ids.includes(parseInt(d['DocId'])));

    // Populate the topic list with given length
    function populateTopicList(topics, max_length, p_div){
        const p = p_div.find('p');
        p.empty();
        for(let i=0; i< max_length; i++) {
            const topic = topics[i];
            const link = $('<button type="button" class="btn btn-link">'
                + topic['topic'] + ' (' + topic['doc_ids'].length + ') </button>');
            link.button();
            link.click(function (event) {
                $('#topics').val(topic['topics']);
                // // Get the documents about the topic
                const doc_list_heading = new DocumentListHeading(cluster_topics, cluster_docs, topic);

            });
            p.append(link);
        }
    }

    // Create a list of topics
    function createTopicParagraphs(topics){
        const p_div = $('<div class="topic_list""><p></p></div>');
        populateTopicList(topics, 30, p_div);   // Display top 30 topics
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
        view_more_btn.click(function(event){
            // Display top 100 topics
            populateTopicList(topics, 100, p_div);
            // Hide view more
            view_more_btn.hide();
            view_less_btn.show();
        });
        // Define view less btn click event
        view_less_btn.click(function (event){
            // Display top 30 topics
            populateTopicList(topics, 30, p_div);
            // Hide view more
            view_more_btn.show();
            view_less_btn.hide();
        });
        return p_div;
    }

    // Create a list view of topics derived by using BERTopic
    function createTopicListView() {
        $('#topic_list_view').empty();
        const container = $('<div class="container"></div>');
        container.append($('<div class="h3">Cluster #' + cluster_no+' has ' + cluster_docs.length + ' articles in total</div>'));

        // Display the topics of 1 gram, 2 grams and 3 grams
        for(const n_gram of ['1-gram', '2-gram', '3-gram', 'N-gram']){
            const topics = cluster_topics['Topic' + n_gram];
            const key_term_div = $('<div><h3><span class="fw-bold">'+ n_gram + ' Topics </span> ' +
                '</h3>' +
                '<div class="topics">' +
                '</div>' +
                '</div>');
            const sort_btn_name = 'sort'+n_gram;
            const sort_widget = $('<div>' +
                '<span>Sort by </span>'+
                '<div class="form-check form-check-inline">' +
                '   <label class="form-check-label" for="score">Score</label>' +
                '   <input class="form-check-input" type="radio" name="'+sort_btn_name + '" value="score" checked>' +
                '</div>' +
                '<div class="form-check form-check-inline">' +
                '   <label class="form-check-label" for="count">Count</label>' +
                '   <input class="form-check-input" type="radio" name="'+sort_btn_name + '" value="count" >' +
                '</div>' +
                '</div>');
            // Sort the topics by count (default)
            topics.sort((a, b) => b['score'] - a['score']);
            // Add a radio button to sort topics by scores or range
            key_term_div.find(".topics").append(sort_widget);
            key_term_div.find(".topics").append(createTopicParagraphs(topics)); // Show topics
            // Add the on click event to radio button
            sort_widget.find('input[name='+sort_btn_name+']').change(function(){
                const sorted_index = sort_widget.find('input[name='+sort_btn_name+']:checked').val();
                const sorted_topic = cluster_topics['Topic' + n_gram];
                if(sorted_index === 'count'){
                    // Sort the topic by count
                    sorted_topic.sort((a, b) => b['doc_ids'].length - a['doc_ids'].length);
                }else{
                    sorted_topic.sort((a, b) => b['score'] - a['score'] );// Sort by score
                }
                // console.log(topics);
                key_term_div.find(".topic_list").remove();
                key_term_div.find(".topics").append(createTopicParagraphs(sorted_topic));
            });

            // Display n-gram topics by default
            if(n_gram === 'N-gram'){
                // Show the topics for TF-IDF
                key_term_div.accordion({
                    collapsible: true,
                    heightStyle: "fill"
                });
            }else{
                key_term_div.accordion({
                    collapsible: true,
                    heightStyle: "fill",
                    active: 2
                });
            }
            container.append($("<div class='row'><div class='col'></div></div>").find(".col").append(key_term_div));
        }
        $('#topic_list_view').append(container);
    }


    function _createUI(){
        $('#topics').empty();   // Clean the topics
        // Initialise the auto-complete topic with all topics
        $( "#topics" ).autocomplete({
            source: available_topics.map(t => t['topic'])
        });
        // Search topic within a cluster
        $('#search').button();
        $('#search').click(function(event){
            const select_topic = $('#topics').val();
            const topic = available_topics.find(t => t['topic'] === select_topic);
            const doc_list_heading = new DocumentListHeading(cluster_topics, cluster_docs, topic);
        });
        // Clear button to clearn search input and display all the articles
        $('#clear').button();
        $('#clear').click(function(event){
            $( "#topics" ).val("");
            // Display all the articles in a cluster
            const doc_list_heading = new DocumentListHeading(cluster_topics, cluster_docs, null);
        });

        // Create a list of topics
        createTopicListView();

        // Display all the articles in a cluster
        const doc_list_heading = new DocumentListHeading(cluster_topics, cluster_docs, null);

    }
    _createUI();
}
