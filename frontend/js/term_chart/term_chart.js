// Draw the term chart and document list
function TermChart(cluster_data, corpus_data){
    const cluster_docs = corpus_data.filter(d => cluster_data['DocIds'].includes(d['DocId']))
    console.log(cluster_docs);
    const lda_topics = cluster_data['LDATopics'];

    function _createUI(){
        try{
            const topic_data = lda_topics[0];
            const network_graph = new D3NetworkGraph(topic_data, cluster_docs);   // Network graph
            // const year_control = new YearControl(searched_term, term_map, documents);
        }catch (error){
            console.error(error);
        }
    }
    _createUI();
}
