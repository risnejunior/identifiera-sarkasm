var ajaxurl = "index.php"
var debug = true

/* event handlers ####################################################
####################################################################*/
$( document ).ready(function() {
	debug ? console.log("document load") : null
	var user_id = null 

	//user unknown
	if (localStorage.getItem('user_id') === null ||
		localStorage.getItem('nonce') === null ) {
		
		$("#intro").show();

	//user known
	} else {
		update_quiz('easy', 10);
	}
});

// register user
$("#name-form").keypress(function(e) {
	if(e.which == 13){
		$("#name-form").submit();
	}
});

$("#name-form").submit(function(event) {
	event.preventDefault();
	register_user();
});

$("li.hard-link").click(function(e){
	update_quiz('hard', 10);
});

$("li.easy-link").click(function(e){
	update_quiz('easy', 10);
});

/* funs #############################################################
####################################################################*/

function send_ajax(message, success_fun) {
	debug ? console.log("send ajax") : null
	var settings = {
	     "url": ajaxurl,
	     "timeout": 5000,
	     "type": "post",
	     "dataType": "json",
	     "data": message,
	     "complete": complete,
	     "error": error,
	     "success": success,
	     "context": this
	};
	 
	 //ajax call bound to deferred
	jQuery.ajax( settings );

	function success( data, textStatus, jqXHR ) {
		
		//serverside error 
	    if ( !data.type || "error" === data.type ) {
	        var err_message = "An error occurred!";
	        if ( 'errors' in data ) {
	            err_message = "Error: " + data.errors.join(',');
	        }
	        jQuery("#status").html( err_message ).show();

	        /*
	    	if (err_message.indexOf("User not found") == -1) {
	    		localstore.clear();
	    		location.reload();
	    	}
	    	*/
	    
	    //server side success: set quiz
	    } else {
	    	success_fun(data);
	    }
	}

	function error( jqXHR, textStatus, errorThrown ) {
		jQuery("#status").html( "Server error!" ).show();
	}

	function complete( jqXHR, textStatus ) {
		jQuery( this )
		    .parent()
		    .siblings( ".animation-container" )
		    .removeClass( "working-animation" );
	}
}

function register_user() {
	debug ? console.log("register user") : null
	var message = {
		'name': $( "input[name*='name']" ).first().val(),
		'action': 'register_user'
	};

	send_ajax(message, success);

	function success(data) {
		localStorage['user_id'] = data.id;
		localStorage['nonce'] = data.nonce;
		localStorage['name'] = data.name;

		update_quiz('easy', 10);
	}
}

function get_quiz(dataset, size, callback) {
	debug ? console.log("get quiz") : null

	// check if quiz stored
	if (!(dataset in localStorage)) {
		debug ? console.log(dataset + " NOT in localstore") : null
	
		var message = {
			'action': 'get_quiz',
			'dataset': dataset,
			'user_id': parseInt(localStorage['user_id']),
			'nonce': localStorage['nonce'],
			'size': size
		};

		send_ajax(message, success);

	} else {
		debug ? console.log(dataset + " IS in localstore") : null
		quiz = JSON.parse(localStorage[dataset]);		
		callback(quiz);
	}


	function success(data) {
		if (data.quiz.length < 1) {
			debug ? console.log("Empty quiz returned!") : null
			location.reload();
		}

		localStorage[data.dataset] = JSON.stringify(data.quiz);
		callback(data.quiz);
	}
}

function add_quiz_html(quiz, container) {
	debug ? console.log("add quiz html to " + container.id) : null
	$(".quiz-container").empty();
	var answers = get_answers();

	for (var i = 0; i < quiz.length; i++) {		
		question = quiz[i];
		//console.log(question.sample_text);

		var sectionElement = $("<section></section>")
			.attr("id", question.id)
			.text(question.sample_text)
			.css("background", "#f8ede8")
			.css("margin", "5px")
			.css("padding", "5px")
			.css("border-radius", "6px");

		var buttonPart = $("<div></div>")
			.addClass("button-part")
			.append("<button class='pos button'>Sarcastic</button>")
			.append("<button class='neg button'>Other</button>");

		if (question.id in answers && answers[question.id] == true) {
			$('.pos', buttonPart).addClass("selected");
		} else if (question.id in answers && answers[question.id] == false) {
			$('.neg', buttonPart).addClass("selected");
		}

		sectionElement.append(buttonPart);
		$(container).append(sectionElement);
	}

	$("button", ".button-part").click(function(event) {
		var question_id = $(this).parents("section").first().attr("id");		
		var isPos = $(this).hasClass("pos");
		
		if ( !$(this).hasClass('selected') ) {
			$(this).addClass('selected');
		}

		$(this).siblings('.button:first').removeClass('selected');

		set_answer(question_id, isPos);
	});

	jQuery("#aside, #navbar").show();	
	jQuery("#intro").hide();
	jQuery("container").show();
	jQuery("#score, h3:first").html(localStorage['name'] + "'s score: ").show();	
}

function set_answer(question_id, isPos) {
	debug ? console.log("Set answer: " + question_id + " " + isPos) : null
	user_id = localStorage.getItem('user_id')
	nonce = localStorage.getItem('nonce')
	var message = {
		'action': 'save_answer',
		'nonce': nonce,
		'question_id': parseInt(question_id),
		'user_id': parseInt(user_id),		
		'answer': isPos
	};

	send_ajax(message, success)

	function success(data) {
		answers = get_answers()
		answers[question_id] = isPos;
    	localStorage['answers'] = JSON.stringify(answers);
	}
}

function get_answers() {
	var answers = {};
	if ('answers' in localStorage) {
		answers = JSON.parse(localStorage['answers']);
	} 
	return answers;
}

function update_quiz(quiz_name, size) {
	debug ? console.log("update_quiz") : null
	//debug!
	size = 3;

	var dataset = null;
	var container = null;

	if (quiz_name == 'hard') {		
		dataset = 'poria-ratio';
		container = $("#hard-quiz");
	} else {
		dataset = 'poria-balanced';
		container = $("#easy-quiz");
	}

	 get_quiz(dataset, size, callback);

	function callback(quiz) {
		debug ? console.log("get quiz callback") : null
		add_quiz_html(quiz, container);
	}
	
}

function tally_score() {
	var total = 0
	var tp = 0, tn = 0, fp = 0, fn = 0

	for (var key in results) { 
		var actual = quiz_key[key]
		var answer = results[key]
		total += 1
		if ( actual == answer && actual) {
			tp += 1;
		} else if (actual == answer && !actual ) {
			tn += 1;
		} else if (actual != answer && actual) {
			fp += 1;
		} else if (actual != answer && !actual) {
			fn += 1;
		} else {
			throw "Should not fall through!";
		}
	}
	var accuracy = total > 0 ? ((tp + tn) / total) : 0
	var precision = (tp + fp) > 0 ? (tp / (tp + fp)) : 0
	var recall = (tp + fn) > 0 ? (tp / (tp + fn)) : 0
	var f1_score =  (precision + recall) > 0 ? 2*((precision * recall) / (precision + recall )) : 0 				
	accuracy = Math.round(accuracy * 100) / 100;
	f1_score = Math.round(f1_score * 100) / 100;

	answer_string = 'Accuracy: ' + accuracy + ', F1 Score: ' + f1_score + " "
	answer_string += ' ,Correct: ' + (tp + tn) + ' out of: ' + total

	console.log('total: '+ total)
}