// Draw the term chart and document list
function TermChart(cluster, cluster_docs){
    let keyword_groups = cluster['KeyPhrases'];
    let total_keywords = 0;
    let frequent_words = [];
    // Populate the topic words of each keyword group
    for(const keyword_group of keyword_groups){
        try {
            const new_topic_words = Utility.get_top_5_words_from_keyword_group(keyword_group);
            keyword_group['topic_words'] = new_topic_words;
            const group_doc_ids = keyword_group['DocIds'];
            const key_phrases = keyword_group['key-phrases'];
            total_keywords += key_phrases.length;
            // // Get group docs
            // const group_docs = cluster_docs.filter(d => group_doc_ids.includes(d['DocId']))
            // const word_key_phrase_dict = Utility.create_word_key_phrases_dict(new_topic_words, key_phrases);
            // const word_doc_dict = Utility.create_word_doc_dict(new_topic_words, group_docs, word_key_phrase_dict);
            // for(const topic_word of new_topic_words){
            //     const count = word_doc_dict[topic_word].length;
            //     if(topic_word  !== "others"){
            //         let found = frequent_words.find(w => w['word'] === topic_word);
            //         if(found){
            //             found['count'] = count;
            //         }else{
            //             frequent_words.push({'word': topic_word, "count": count})
            //         }
            //     }
            // }
        } catch (error) {
            console.error(error);
        }
    }
    // // Sort the frequent_words by count
    // frequent_words.sort((a, b) =>{
    //     return b['count'] - a['count'];
    // });
    // console.log("article cluster", cluster['Cluster'], ": ", frequent_words.slice(0, 10).map(w => w['word']).join(", "));


    // Create a cluster heading
    function createClusterHeading(){
        const cluster_no = cluster['Cluster'];
        const heading = $('<div class="mt-3"><span class="h5">Article Cluster ' + cluster_no +'</span></div>');
        const cluster_link = $('<span type="button" class="btn btn-link" >' + cluster_docs.length + ' articles</span>');
        // Define onclick event cluster link
        cluster_link.click(function(){
            const doc_list = new DocList(cluster_docs, [], "");
            document.getElementById('doc_list').scrollIntoView({behavior: "smooth",
                block: "nearest", inline: "nearest"});
        });
        heading.append(cluster_link);
        heading.append($('<span>' + total_keywords + ' keywords</span>'));

        $('#cluster_heading').empty();
        $('#cluster_heading').append(heading);
    }


    // Main entry
    function createUI(){
        try{
            $('#term_occ_chart').empty();
            $('#sub_group').empty();
            $('#doc_list').empty();
            $('#key_phrase_chart').empty();
            // Display bar chart to show groups of key phrases as default graph
            const graph = new BarChart(keyword_groups, cluster, cluster_docs);
            createClusterHeading();
        }catch (error){
            console.error(error);
        }
    }
    createUI();
}
