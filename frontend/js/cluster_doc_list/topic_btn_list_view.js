// Create an auto-complete
function TopicBtnListView(cluster_no, cluster_topic_words, doc_key_terms){
    // Fill out the topic using TF-IDF
    const cluster_topics = cluster_topic_words.find(c => c['Cluster'] === cluster_no);
    const n_gram_topics = cluster_topics['Topic1Grams'].concat(cluster_topics['Topic2Grams'], cluster_topics['Topic3Grams']);
    // Sort the topics by scores
    n_gram_topics.sort((a, b) => b['score'] - a['score']);
    const available_topics = n_gram_topics.map(t => t['topic']);
    const cluster_doc_ids = cluster_topics['DocIds'];
    const cluster_docs = doc_key_terms.filter(d => cluster_doc_ids.includes(parseInt(d['DocId'])));


    // Create a list of topics
    function createTopicParagraphs(topics){
        const p_div = $('<p class="topic_list"></p>');
        for(let i=0; i< topics.length; i++){
            const topic = topics[i];
            const link = $('<button type="button" class="btn btn-link">'
                + topic['topic'] + ' (' + topic['doc_ids'].length + ') </button>');
            link.button();
            link.click(function(event){
                $('#topics').val(topic['topics']);
                // // Get the documents about the topic
                const topic_docs = doc_key_terms.filter(d => topic['doc_ids'].includes(parseInt(d['DocId'])));
                const doc_list_view = new DocumentListView(cluster_topics, topic_docs, topic);
            });
            p_div.append(link);
        }
        return p_div;
    }

    // Create a list view of topics derived by using BERTopic
    function createTopicListView() {
        $('#topic_list_view').empty();
        const container = $('<div class="container"></div>');
        const display_names = {'1Grams': '1-gram', '2Grams': '2-gram', '3Grams': '3-gram', 'N-gram': 'N-gram'};
        // Display the topics of 1 gram, 2 grams and 3 grams
        for(const n_gram of ['1Grams', '2Grams', '3Grams', 'N-gram']){
            const topics = n_gram !== 'N-gram' ? cluster_topics['Topic' + n_gram] : n_gram_topics;
            const key_term_div = $('<div><h3><span class="fw-bold">'+ display_names[n_gram] + ' Topics </span> ' +
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
                '   <input class="form-check-input" type="radio" name="'+sort_btn_name + '" value="count">' +
                '</div>' +
                '</div>');
            // Add a radio button to sort topics by scores or range
            key_term_div.find(".topics").append(sort_widget);
            key_term_div.find(".topics").append(createTopicParagraphs(topics.slice(0, 50))); // Show top 50 topics
            // Add the on click event to radio button
            sort_widget.find('input[name='+sort_btn_name+']').change(function(){
                const sorted_index = sort_widget.find('input[name='+sort_btn_name+']:checked').val();
                const sorted_topic = n_gram !== 'N-gram' ? cluster_topics['Topic' + n_gram] : n_gram_topics;
                if(sorted_index === 'count'){
                    // Sort the topic by count
                    sorted_topic.sort((a, b) => b['doc_ids'].length - a['doc_ids'].length );
                }else{
                    sorted_topic.sort((a, b) => b['score'] - a['score'] );// Sort by score
                }
                // console.log(topics);
                key_term_div.find(".topic_list").remove();
                key_term_div.find(".topics").append(createTopicParagraphs(sorted_topic.slice(0, 50)));
            });


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
        $( "#topics" ).autocomplete({
            source: available_topics
        });
        // Search topic within a cluster
        $('#search').button();
        $('#search').click(function(event){
            const select_topic = $('#topics').val();
            const topic = n_gram_topics.find(t => t['topic'] === select_topic);
            // Get the articles related to the selected topic.
            const topic_docs = doc_key_terms.filter(d => topic['doc_ids'].includes(parseInt(d['DocId'])));
            const doc_list_view = new DocumentListView(cluster_topics, topic_docs, topic);
        });
        // Clear button to clearn search input and display all the articles
        $('#clear').button();
        $('#clear').click(function(event){
            $( "#topics" ).val("")
            // Display all the articles in a cluster
            const doc_list_view = new DocumentListView(cluster_topics, cluster_docs, null);
        });

        // Create a list of topics
        createTopicListView();

        // Display all the articles in a cluster
        const doc_list_view = new DocumentListView(cluster_topics, cluster_docs, null);
    }
    _createUI();
}
