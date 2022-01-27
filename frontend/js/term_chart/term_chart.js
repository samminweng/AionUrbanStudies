// Draw the term chart and document list
function TermChart(cluster_data, cluster_docs){
    const lda_topics = cluster_data['LDATopics'];
    const key_phrases = cluster_data['KeyPhrases'];
    // Initailise the group list
    function create_group_list(){
        $('#group_list').empty();
        const select_menu = $('<select name="item"></select>');
        // Add a list of LDA topics
        for(let i=0; i< lda_topics.length; i++){
            if(i === 0){
                select_menu.append($('<option value="topic_' + i +'" selected> LDA Topics #' + i +' </option>'));
            }else{
                select_menu.append($('<option value="topic_' + i +'"> LDA Topics #' + i +' </option>'));
            }
        }
        // Add a list of
        for(let i=0; i< key_phrases.length; i++){
            select_menu.append($('<option value="phrase_' + i +'"> Phrase Group #' + i +' </option>'));
        }
        $('#group_list').append(select_menu);
        //
        select_menu.selectmenu({
            change: function( event, data ) {
                // console.log( data.item.value);
                const item = data.item.value;
                const is_key_phrase = item.split("_")[0] === 'phrase';
                const index = parseInt(item.split("_")[1]);
                let selected_data = lda_topics[index];
                if(is_key_phrase){
                    selected_data = key_phrases[index];
                }
                const word_docs = selected_data['word_docIds'];
                $('#score').text(selected_data['score'].toFixed(3));
                const network_graph = new D3NetworkGraph(word_docs,  cluster_docs, is_key_phrase);
            }
        });
        select_menu.selectmenu("refresh");
    }
    // Main entry
    function createUI(){
        try{
            create_group_list();
            const word_docs = lda_topics[0]['word_docIds'];
            $('#score').text(lda_topics[0]['score'].toFixed(3));      // Update the score
            // Network graph
            const network_graph = new D3NetworkGraph(word_docs,  cluster_docs, false);
        }catch (error){
            console.error(error);
        }
    }
    createUI();
}
