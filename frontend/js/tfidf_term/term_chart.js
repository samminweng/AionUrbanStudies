// Draw the term chart and document list
function TermChart(searched_term, collocation_data, doc_term_data){
    const collocation = collocation_data.find(({Collocation}) => Collocation === searched_term);
    // console.log(collocation);
    const documents = Utility.collect_documents_by_doc_ids(collocation, doc_term_data);
    // console.log(documents);
    const term_map = doc_term_data.find({});
    console.log(term_map);

    function _createUI(){

    }

    _createUI();
}
