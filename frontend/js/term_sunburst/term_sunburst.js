// Draw the term chart and document list
function TermSunburst(cluster, cluster_docs){
    const lda_topics = cluster['LDATopics'];
    const groups = cluster['KeyPhrases'];
    const sub_groups = cluster['SubGroups'];
    const phrase_total = groups.reduce((pre, cur) => pre + cur['NumPhrases'], 0);
    // Create the header
    function create_header(){
        $('#paper_count').text(cluster['NumDocs']);
        $('#phrase_count').text(groups.length);
        $('#phrase_total').text(phrase_total);
    }

    // Main entry
    function createUI(){
        try{
            // Display key phrase groups as default graph
            const sunburst_graph = new SunburstGraph(groups, sub_groups, cluster, cluster_docs);
            // create_header();
        }catch (error){
            console.error(error);
        }
    }
    createUI();
}
