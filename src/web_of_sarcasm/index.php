<?php
	require  'util.php';

	$errors = new Errors();

	if (isset($_POST['action'])) {

		$db = new DbAdapter();
		$action = $_POST['action'];

		if ($action == 'get_quiz') {			
			$user_id = intval($_POST['user_id']);
			$nonce = $_POST['nonce'];
            $size = intval($_POST['size']);
			$dataset = $_POST['dataset'];

			if ($db->validate_user($user_id, $nonce)) {
				$quiz = $db->getQuiz($size, $dataset);	
				$result_array = array(
                    'type'=>'set_quiz',
					'dataset'=>$dataset,
					'quiz' =>$quiz
				);
                if (count($quiz) < 1) {
                    $errors->add("No quiz found");
                }
			} else {
				$errors->add("user not validated");
			}			

		} elseif ($action == 'register_user') {
			$name = trim($_POST['name']);
						
            if (strlen($name) > 30) {
                $errors->add("username too long, should be < 30 charachters.");
            } else if (strlen($name) < 2) {
                $errors->add("username too short, should be at least 1 charachter.");
            } else {
                $user = $db->createUser($name);
                $result_array = array(
                    'type'=>'set_user',
                    'id'=>$user['id'],
                    'nonce'=>$user['nonce'],
                    'name'=>$user['name']
                );
            }            	

		} elseif ($action == 'save_answer') {            
            $question_id = intval($_POST['question_id']);
            $user_id = intval($_POST['user_id']);
            $nonce = $_POST['nonce'];
            $answer = intval($_POST['answer']);

            if ($db->validate_user($user_id, $nonce)) {

                $db->setAnswer($question_id, $user_id, $answer);  
                
                $result_array = array(
                    'type'=>'saved',
                    'question_id'=>$question_id,
                    'answer'=>$answer
                );
            } else {
                $errors->add("user not validated");
            }           
		} else if ($action == 'tally_score') {
            $user_id = intval($_POST['user_id']);
            $dataset = $_POST['dataset'];
            $score = $db->getScore($user_id, $dataset);
            $result_array = array(
                'type'=>'score_tally',
                'tally'=>$score,
                'dataset'=>$dataset
            );            
        } else {
			$errors->add("Request command not recognized");
		}

		if ($errors->has() || $db->errors->has()) {
			$all_errors = array_merge($errors->getAll(), $db->errors->getAll());
			$result_array = array(
				'type'=>'error',
				'errors'=>$all_errors
			);

            $date = date('Y-m-d H:i:s');
            error_log( $date . 
                        ': ' . 
                        implode(', ', $all_errors) . 
                        "\n", 3, "errors.log"
            );
		} else {
            //
        }

		echo json_encode( $result_array );
		die();
	} 
?>

<!doctype html>
<!--[if lt IE 7]>      <html class="no-js lt-ie9 lt-ie8 lt-ie7" lang=""> <![endif]-->
<!--[if IE 7]>         <html class="no-js lt-ie9 lt-ie8" lang=""> <![endif]-->
<!--[if IE 8]>         <html class="no-js lt-ie9" lang=""> <![endif]-->
<!--[if gt IE 8]><!--> <html class="no-js" lang=""> <!--<![endif]-->
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
        <title></title>
        <meta name="description" content="">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="apple-touch-icon" href="apple-touch-icon.png">

        <link rel="stylesheet" href="css/normalize.min.css">
        <link rel="stylesheet" href="css/main.css">

        <script src="js/vendor/modernizr-2.8.3-respond-1.4.2.min.js"></script>
    </head>
    <body>
        <!--[if lt IE 8]>
            <p class="browserupgrade">You are using an <strong>outdated</strong> browser. Please <a href="http://browsehappy.com/">upgrade your browser</a> to improve your experience.</p>
        <![endif]-->

        <div class="header-container">
            <header class="wrapper clearfix">
                <h1 class="title">Sarcasm quiz!</h1>
                <nav id='navbar' hidden=true>
                    <ul>
                        <li id='hard-link'><a href="#">Hard quiz</a></li>
                        <li id='easy-link' class='selected'><a href="#">Easy quiz</a></li>
                    </ul>
                </nav>
            </header>
        </div>

        <div class="main-container">
            <div id="main-content" class="main wrapper clearfix">

            	<article id='intro' hidden=true>
            	    <header id='content-header'>
            	        <h3>Please provide your name</h3>
            	        <form id='name-form'>
            	        	<input type='text' name='name'><br>
            	        	<input type='submit' value='Submit' id='submit-name'>
            	        </form>
            	    </header>        	    
            	</article>


                <article id='quiz' class='quiz-container easy'>    
                </article>        

                <aside id='aside' hidden='true'>
                	<div id='status'></div>
                    <div class='tooltip'>
                    <ul id='score'>
                        <li>Answered: <div id='count' class='metric'>0</div></li>
                        <li>Accuracy: <div id='accuracy' class='metric'>0</div></li>
                        <li>Precision: <div id='precision' class='metric'>0</div></li>
                        <li>Recall: <div id='recall' class='metric'>0</div></li>
                        <li>F1-score: <div id='f1_score' class='metric'>0</div></li>
                    </ul>                                                    
                        <span class="tooltiptext">Your score is updated when you've answered all visable questions
                        </span>                        
                    </div>
                </aside>

            </div> <!-- #main -->
        </div> <!-- #main-container -->

        <div class="footer-container">
            <footer class="wrapper">
                <h3>footer</h3>
            </footer>
        </div>

        <script src="js/vendor/jquery-1.11.2.js"></script>
        <script src="js/main.js"></script>

        <!--
        <script src="//ajax.googleapis.com/ajax/libs/jquery/1.11.2/jquery.js"></script>
        <script>window.jQuery || document.write('<script src="js/vendor/jquery-1.11.2.js"><\/script>')</script>

        

         Google Analytics: change UA-XXXXX-X to be your site's ID. 
        <script>
            (function(b,o,i,l,e,r){b.GoogleAnalyticsObject=l;b[l]||(b[l]=
            function(){(b[l].q=b[l].q||[]).push(arguments)});b[l].l=+new Date;
            e=o.createElement(i);r=o.getElementsByTagName(i)[0];
            e.src='//www.google-analytics.com/analytics.js';
            r.parentNode.insertBefore(e,r)}(window,document,'script','ga'));
            ga('create','UA-XXXXX-X','auto');ga('send','pageview');
        </script>
        -->
    </body>
</html>

