// Draw the term chart and document list
function TermSunburst(cluster_data, cluster_docs){
    const lda_topics = cluster_data['LDATopics'];
    const key_phrase_groups = cluster_data['KeyPhrases'];
    // // Initailise the group list
    // function create_group_list(){
    //     $('#group_types').empty();
    //     const select_menu = $('<select name="item"></select>');
    //     // Add shared phrase groups and
    //     select_menu.append($('<option value="phrases" selected>Phrase Groups</option>'));
    //     select_menu.append($('<option value="topics">LDA Topics</option>'));
    //
    //     $('#group_types').append(select_menu);
    //     //
    //     select_menu.selectmenu({
    //         change: function( event, data ) {
    //             // console.log( data.item.value);
    //             const item = data.item.value;
    //             const is_key_phrase = item === 'phrases';
    //             let selected_data = lda_topics;
    //             if(is_key_phrase){
    //                 selected_data = key_phrases;
    //             }
    //             console.log(selected_data);
    //             // const word_docs = selected_data['word_docIds'];
    //             // $('#score').text(selected_data['score'].toFixed(3));
    //             // const graph = new D3NetworkGraph(word_docs,  cluster_docs, is_key_phrase);
    //         }
    //     });
    //     select_menu.selectmenu("refresh");
    // }

    // Main entry
    function createUI(){
        try{
            const cluster_no = cluster_data['Cluster'];
            // Display key phrase groups as default graph
            const graph = new SunburstGraph(key_phrase_groups, cluster_no, cluster_docs);
        }catch (error){
            console.error(error);
        }
    }
    createUI();
}
