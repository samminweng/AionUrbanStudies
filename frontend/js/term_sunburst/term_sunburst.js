// Draw the term chart and document list
function TermSunburst(cluster_data, cluster_docs){
    const lda_topics = cluster_data['LDATopics'];
    const key_phrase_groups = cluster_data['KeyPhrases'];


    // Main entry
    function createUI(){
        try{
            const cluster_no = cluster_data['Cluster'];
            // Display key phrase groups as default graph
            const key_phrase_graph = new SunburstGraph(key_phrase_groups, cluster_no, true);
            const lad_topic_graph = new SunburstGraph(lda_topics, cluster_no, false);
        }catch (error){
            console.error(error);
        }
    }
    createUI();
}
