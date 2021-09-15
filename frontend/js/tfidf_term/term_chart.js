// Draw the term chart and document list
function TermChart(searched_term, collocation_data, doc_term_data){
    function _createUI(){
        try{
            const collocation = collocation_data.find(({Collocation}) => Collocation === searched_term);
            const documents = TermChartUtility.collect_documents_by_doc_ids(collocation, doc_term_data);
            const term_map = collocation['TermMap'];
            const network_graph = new D3NetworkGraph(searched_term, term_map, documents);   // Network graph
            const year_control = new YearControl(searched_term, term_map, documents);
        }catch (error){
            console.error(error);
        }
    }
    _createUI();
}
