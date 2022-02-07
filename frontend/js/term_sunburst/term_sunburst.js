// Draw the term chart and document list
function TermSunburst(cluster_data, cluster_docs){
    const lda_topics = cluster_data['LDATopics'];
    const groups = cluster_data['KeyPhrases'];
    const sub_groups = cluster_data['SubGroups'];
    const phrase_total = groups.reduce((pre, cur) => pre + cur['NumPhrases'], 0);
    // Create the header
    function create_header(){
        $('#paper_count').text(cluster_data['NumDocs']);
        $('#phrase_count').text(groups.length);
        $('#phrase_total').text(phrase_total);
    }

    // Main entry
    function createUI(){
        try{
            const cluster_no = cluster_data['Cluster'];
            // Display key phrase groups as default graph
            const sunburst_graph = new SunburstGraph(groups, sub_groups, cluster_no, true);
            create_header();
        }catch (error){
            console.error(error);
        }
    }
    createUI();
}
