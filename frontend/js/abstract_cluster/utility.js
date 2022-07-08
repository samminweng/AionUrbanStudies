// Utility functions
class Utility {

    // Collect the unique top 3 terms
    static get_top_terms(cluster_terms, n) {
        // Get top 3 term that do not overlap
        let top_terms = cluster_terms.slice(0, 1);
        let index = 1;
        while (top_terms.length < n && index < cluster_terms.length) {
            const candidate_term = cluster_terms[index];
            try {
                const c_term_words = candidate_term.split(" ");
                if (c_term_words.length >= 2) {
                    // Check if candidate_term overlap with existing top term
                    const overlap = top_terms.find(top_term => {
                        // Check if any cluster terms
                        const top_term_words = top_term.split(" ");
                        // Check two terms are different
                        // Split the terms into words and check if two term has overlapping word
                        // Check if two word list has intersection
                        const intersection = top_term_words.filter(term => c_term_words.includes(term));
                        if (intersection.length > 0) {
                            return true;
                        }
                        return false;
                    });
                    if (typeof overlap !== 'undefined') {
                        try {
                            if (overlap.split(" ").length !== candidate_term.split(" ").length) {
                                const overlap_index = top_terms.findIndex(t => t === overlap);
                                // Replace existing top term with candidate
                                top_terms[overlap_index] = candidate_term;
                            }// Otherwise, skip the term
                        } catch (error) {
                            console.error(error);
                            break;
                        }
                    } else {
                        top_terms.push(candidate_term);
                    }
                }
            } catch (error) {
                console.error(error);
                break;
            }
            index = index + 1;
        }
        // Check if the top_terms has n terms
        while (top_terms.length < n) {
            // Get the last term
            const last_term = top_terms[top_terms.length - 1];
            const last_term_index = cluster_terms.findIndex(t => t === last_term) + 1;
            top_terms.push(cluster_terms[last_term_index]);
        }

        return top_terms;
    }


}


