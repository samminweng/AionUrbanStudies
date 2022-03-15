// Create a div to display the topics obtained by LDA topic model
function LDATopicView(lda_topics, accordion_div){
    // console.log(cluster_lda_topics);
    // Create an list item to display a group of key phrases
    function createLDATopicView(index, lda_topic){
        // const index = lda_topic['index'];
        const topic_view = $('<li class="list-group-item d-flex justify-content-between align-items-start"></li>')
        // Display key phrases
        const topic_words = lda_topic['topic_words'];
        const topic_word_div = $('<div class="ms-2 me-auto"> <div class="fw-bold">Topic #' + (index + 1) +'</div> </div>');
        // Display top 10 key phrases
        const topic_word_span = $('<p class="lda_topic_text"></p>');
        topic_word_span.text(topic_words.join(", "));
        topic_word_div.append(topic_word_span);
        topic_view.append(topic_word_div);
        return topic_view;
    }


    // Create a pagination to show the LDA topics by pages
    function create_view() {
        // Add the table header
        const topic_view_list = $('<ul class="list-group list-group-flush"></ul>');
        for (let i = 0; i < lda_topics.length; i++) {
            const lda_topic = lda_topics[i];
            console.log(lda_topic);
            const lda_topic_views = createLDATopicView(i, lda_topic);
            topic_view_list.append(lda_topic_views);
        }

        return topic_view_list;
    }


    function _createUI(){
        $('#lda_topic_view').empty();
        // Heading
        const heading = $('<h3>LDA Topic Model</h3>');
        const p = $('<p></p>');
        // A list of grouped key phrases
        const view = create_view();
        p.append(heading);
        p.append(view);
        $('#lda_topic_view').append(p);
    }


    _createUI();

}
