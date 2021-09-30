// Create an auto-complete
function TopicBtnListView(cluster_no, cluster_topic_words, doc_key_terms){
    // Fill out the topic using TF-IDF
    const cluster_topics = cluster_topic_words.find(c => c['Cluster'] === cluster_no);
    const topic_words = cluster_topics['TopicWords_by_TF-IDF'];
    const available_topics = topic_words.map(t => t['topic_words']);
    const cluster_doc_ids = cluster_topics['DocIds'];
    const cluster_docs = doc_key_terms.filter(d => cluster_doc_ids.includes(parseInt(d['DocId'])));

    // Create a list of topics
    function createTopicParagraphs(topics){
        const p_div = $('<p></p>');
        for(let i=0; i< topics.length; i++){
            const topic = topics[i];
            const link = $('<button type="button" class="btn btn-link">'
                + topic['topic_words'] + ' (' + topic['doc_ids'].length + ') </button>');
            link.button();
            link.click(function(event){
                $('#topics').val(topic['topic_words']);
                // // Get the documents about the topic
                const topic_docs = doc_key_terms.filter(d => topic['doc_ids'].includes(parseInt(d['DocId'])));
                const doc_list_view = new DocumentListView(cluster_topics, topic_docs, [topic['topic_words']]);
            });
            p_div.append(link);
        }
        return p_div;
    }

    // Create a list view of topics derived by three approaches (TF-IDF, Topic2Vec, Collocations)
    function createTopicListView() {
        $('#topic_list_view').empty();
        // Display the topics derived by TF-IDF, Topic2Vec and Collocations
        for(const approach of ['TF-IDF', 'Topic2Vec', 'Collocation']){
            const topics = cluster_topics['TopicWords_by_' + approach];
            const key_term_div = $('<div><h3><span class="fw-bold">Topics by '+ approach + ': </span></h3>' +
                '<div class="topic_words"></div>' +
                '</div>');
            key_term_div.find(".topic_words").append(createTopicParagraphs(topics));
            if(approach === 'TF-IDF'){
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
            $('#topic_list_view').append(key_term_div);
        }
    }


    function _createUI(){
        $('#topics').empty();   // Clean the topics
        $( "#topics" ).autocomplete({
            source: available_topics
        });
        // Search topic within a cluster
        $('#search').button();
        $('#search').click(function(event){
            const select_topic = $('#topics').val();
            const topic = topic_words.find(t => t['topic_words'] === select_topic);
            // Get the articles related to the selected topic.
            const topic_docs = doc_key_terms.filter(d => topic['doc_ids'].includes(parseInt(d['DocId'])));
            const doc_list_view = new DocumentListView(cluster_topics, topic_docs, [topic['topic_words']]);
        });
        // Clear button to clearn search input and display all the articles
        $('#clear').button();
        $('#clear').click(function(event){
            $( "#topics" ).val("")
            // Display all the articles in a cluster
            const doc_list_view = new DocumentListView(cluster_topics, cluster_docs, []);
        });

        // Create a list of topics
        createTopicListView();

        // Display all the articles in a cluster
        const doc_list_view = new DocumentListView(cluster_topics, cluster_docs, []);
    }
    _createUI();
}
