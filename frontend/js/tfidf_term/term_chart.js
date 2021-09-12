// Draw the term chart and document list
function TermChart(searched_term, collocation_data, doc_term_data){
    const collocation = collocation_data.find(({Collocation}) => Collocation === searched_term);
    const documents = TermChartUtility.collect_documents_by_doc_ids(collocation, doc_term_data);
    const term_map = collocation['TermMap'];
    const occurrences = collocation['Occurrences'];

    function _createUI(){
        let network_graph = new D3NetworkGraph(searched_term, term_map, occurrences);   // Network graph
        let doc_list_view = new DocumentListView([searched_term], collocation_data, documents);
        let year_control = new YearControl(searched_term, documents, term_map, occurrences);
    }

    _createUI();
}
