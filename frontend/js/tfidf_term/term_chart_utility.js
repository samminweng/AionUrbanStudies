class TermChartUtility {
    // Collect all the document
    static collect_documents_by_doc_ids(collocation, doc_term_data){
        const doc_ids = Object.values(collocation['DocIDs']).flat();
        // console.log(doc_ids);
        const documents = doc_term_data.filter(doc_term => doc_ids.includes(doc_term['DocId']));
        documents.sort((a, b) => b['Year'] - a['Year']);
        return documents;

    }
    // Get the maximal doc length
    static get_max_node_size(nodes){
        let max_node_size = 0;
        for(let node of nodes){
            const doc_ids = node['doc_ids'];
            max_node_size = Math.max(max_node_size, doc_ids.length);
        }
        return max_node_size;
    }
    // Convert the collocations json to the format of D3 network graph
    static create_node_link_data(searched_term, term_map, occurrences){
        // Populate the nodes with collocation data
        let nodes = [];
        for(let i = 0; i < term_map.length; i++){
            const tm = term_map[i];
            const col_name = tm[0];
            const doc_ids = tm[1];
            const group = (col_name !== searched_term ? 1: 0)
            const node = {'id': i, 'name': col_name, 'doc_ids': doc_ids, 'group': group};

            nodes.push(node);
        }
        // Populate the links with occurrences
        let links = [];
        for (let source=0; source < nodes.length; source++) {
            for (let target =0; target < nodes.length; target++){
                if(source !== target){
                    const occ = occurrences[source][target];
                    if(occ.length > 0){
                        // Add the link
                        links.push({source: source, target: target, value: occ.length});
                    }
                }
            }
        }
        // Return
        return {nodes: nodes, links: links};
    }

    // Filter term map by the year
    static filter_term_map(year, documents, term_map){
        let filter_doc_ids = documents.filter(doc => doc['Year'] <= year).map(doc => doc['DocId']);
        console.log(filter_doc_ids);
        let filtered_term_map = [];
        // Filter out term map
        for (const tm of term_map){
            const collocation = tm[0];
            const filtered_doc_ids = tm[1].filter(doc_id => filter_doc_ids.includes(doc_id));
            if(filtered_doc_ids.length > 0){
                filtered_term_map.push([collocation, filtered_doc_ids]);
            }
        }
        console.log(filtered_term_map);
        // Reproduce the occurrence matrix
        let filtered_occurrences = [];
        for(let i = 0; i < filtered_term_map.length; i ++){
            let occ_i = [];
            for(let j=0; j < filtered_term_map.length; j++){
                if(i === j){
                    occ_i.push([]);
                }else{
                    let doc_id_i = new Set(filtered_term_map[i][1]);
                    let doc_id_j = new Set(filtered_term_map[j][1]);
                    // Find the intersected doc
                    let doc_id_i_j = new Set([...doc_id_i].filter(doc_id => doc_id_j.has(doc_id)));
                    doc_id_i_j = Array.from(doc_id_i_j);
                    doc_id_i_j.sort((a, b) => a - b);
                    occ_i.push(doc_id_i_j);
                }
            }
            filtered_occurrences.push(occ_i);
        }
        console.log(filtered_occurrences);
        return { filtered_term_map: filtered_term_map, filtered_occurrences: filtered_occurrences}
    }
    // Filter the documents by key terms
    static filter_documents_by_key_terms(searched_term, complementary_terms, term_map, doc_term_data){
        // console.log(term_map);
        let doc_id_s = term_map.find(term => term[0] === searched_term)[1];
        let intersection = new Set(doc_id_s);
        for(let complementary_term of complementary_terms){
            let doc_id_c = term_map.find(term => term[0] === complementary_term)[1];
            intersection = new Set([...intersection].filter(doc_id => doc_id_c.includes(doc_id)));
        }
        console.log(intersection);
        const filtered_documents = doc_term_data.filter(doc => intersection.has(doc['DocId']));
        // Sort the document by year
        filtered_documents.sort((a, b) => b['Year'] - a['Year']);
        return filtered_documents;
    }


}



