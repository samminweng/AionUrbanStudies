// Draw the term chart and document list
function TermSunburst(cluster_data, cluster_docs){
    const lda_topics = cluster_data['LDATopics'];
    const groups = cluster_data['KeyPhrases'];
    const sub_groups = cluster_data['SubGroups']

    // Main entry
    function createUI(){
        try{
            const cluster_no = cluster_data['Cluster'];
            // Display key phrase groups as default graph
            const sunburst_graph = new SunburstGraph(groups, sub_groups, cluster_no, true);
            // const lad_topic_graph = new SunburstGraph(lda_topics, cluster_no, false);
        }catch (error){
            console.error(error);
        }
    }
    createUI();
}
