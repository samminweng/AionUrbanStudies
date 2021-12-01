function ProgressBar(){
    // Add the progress bar
    function _createProgressBar(){
        // Update the progress bar asynchronously
        $('#progressbar').progressbar({
            value: 0,
            complete: function() {
                $( ".progress-label" ).text( "Complete!" );
            }
        });
        let counter = 0;
        (function asyncLoop() {
            $('#progressbar').progressbar("value", counter++);
            if (counter <= 100) {
                setTimeout(asyncLoop, 100);
            }
        })();
    }
    
    _createProgressBar();

}
