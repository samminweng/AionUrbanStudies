// Draw the term chart and document list
function TermChart(cluster, cluster_docs){
    const lda_topics = cluster['LDATopics'];
    const groups = cluster['KeyPhrases'];
    // const sub_groups = cluster['SubGroups'];
    // console.log("group of key phrases", groups);

    // Main entry
    function createUI(){
        try{
            $('#term_occ_chart').empty();
            $('#sub_group').empty();
            $('#doc_list').empty();
            $('#key_phrase_chart').empty();
            // Display bar chart to show groups of key phrases as default graph
            const graph = new BarChart(groups, cluster, cluster_docs);
            const view = new LDATopicView(lda_topics)

        }catch (error){
            console.error(error);
        }
    }
    createUI();
}
