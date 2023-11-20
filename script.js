function showResults() {
    var company = document.getElementById("company").value;
    var representation = document.getElementById("representation").value;
    var imagePath = 'results/' + company + '_' + representation + '.png';
    document.getElementById("results").innerHTML = '<img src="' + imagePath + '" alt="Result Image">';
}

function showTopSources() {
    var company = document.getElementById("company").value;
    var imagePath = 'results/' + company + '_news' + '.png';
    document.getElementById("results").innerHTML = '<img src="' + imagePath + '" alt="Result Image">';
}
