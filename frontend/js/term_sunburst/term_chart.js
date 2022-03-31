// Draw the term chart and document list
function TermChart(cluster, cluster_docs){
    let keyword_groups = cluster['KeyPhrases'];
    let total_keywords = 0;
    // Populate the topic words of each keyword group
    for(const keyword_group of keyword_groups){
        console.log(keyword_group);
        // const topic_words = keyword_groups['topic_words'];
        const new_topic_words = Utility.get_top_5_words_from_keyword_group(keyword_group);
        // console.log("topic_words:", topic_words);
        // console.log("new topic words:", new_topic_words);
        keyword_group['topic_words'] = new_topic_words;
        total_keywords += keyword_group['key-phrases'].length;
    }

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
        // Add an keyword tag to
        const keyword_div = $('<div class="ui-widget">' +
            '<label for="tags">Keywords: </label>' +
            '<input></div>');
        heading.append(keyword_div);
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
