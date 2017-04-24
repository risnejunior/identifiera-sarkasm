<?php

require "db_settings.php";

class Errors {
	protected $errors = array();

	public function add($error) {
		array_push($this->errors, $error);
	}

	public function getAll() {
		return $this->errors;
	}

	public function has() {
		$hasErrors = false;
		if ($this->errors) {
			$hasErrors = true;
		}

		return $hasErrors;
	}
}

class DbAdapter {
	public $errors;
	private $conn;

	function __construct() {
		$this->errors = new Errors();
		
		// Create connection
		$this->conn = new mysqli(
			DB_SERVER, 
			DB_USER, 
			DB_PASSWORD, 
			DB_NAME
		);

		// Check connection
		if ($this->conn->connect_error) {
		    $this->errors->add($conn->connect_error);
		} 		

		$this->conn->set_charset("utf8");
	}

	function __destruct() {
		$this->conn->close();
	}

	public function validate_user($user_id, $nonce) {
		$user = $this->getUser($user_id);
		$validated = false;
		
		if ($user && $user['nonce'] == $nonce) {
			$validated = true;
		} else {
			$this->errors->add('nonce not matched');
		}

		return $validated;
	}

	public function getScore($user_id, $dataset) {		
		$sql = "
		SELECT 
		u.user_id,
		COUNT(*) answer_count,
		SUM(CASE WHEN a.answer = 1 AND s.class = 1 THEN 1 ELSE 0 END) tp,
		SUM(CASE WHEN a.answer = 0 AND s.class = 0 THEN 1 ELSE 0 END) tn,
		SUM(CASE WHEN a.answer = 1 AND s.class = 0 THEN 1 ELSE 0 END) fp,
		SUM(CASE WHEN a.answer = 0 AND s.class = 1 THEN 1 ELSE 0 END) fn
		FROM users u

		LEFT JOIN answers a
		ON u.user_id = a.user_id

		LEFT JOIN samples s
		ON a.question_id = s.id

		WHERE u.user_id = {$user_id}
		AND s.dataset = '{$dataset}'
		GROUP BY u.user_id
		";

		$rows = $this->execQuery($sql);
		return $rows;		
	}

	public function getQuiz($size, $dataset) {
		$sql = "
		SELECT id, sample_text
		FROM samples AS s
		JOIN ( 
			SELECT (
		  		SELECT CEIL( RAND( ) * id_norm.c + id_norm.m ) 
		  		FROM (
		    		SELECT COUNT( id ) c, MIN( id ) m
		    		FROM samples
		    		WHERE dataset = '{$dataset}'
					) AS id_norm
		  		) AS ids
			) AS rnd
		WHERE s.id >= rnd.ids
		ORDER BY s.id ASC 
		LIMIT {$size}
		 ";
		 $rows = $this->execQuery($sql);

		 return $rows;
	}

	public function setAnswer($question_id, $user_id, $answer) {
		$sql = "
		REPLACE INTO `answers`(`question_id`, `user_id`, `answer`) 
		VALUES ({$question_id}, {$user_id}, {$answer})
		 ";
		 $this->execQuery($sql);
	}

	public function getUser($user_id) {
		$user = NULL;
		if (!is_int($user_id)) {
			$this->errors->add('User id has to be an int');
		} else {

			$sql = "SELECT user_id, name, nonce FROM users WHERE user_id = {$user_id}";
			$rows = $this->execQuery($sql);
			$user = NULL;

			if (count($rows) == 1) {
				$user = $rows[0];
			} else {
				$this->errors->add('User not found');
			}
		}

		return $user;
	}

	public function createUser($user_name) {
		$nonce = self::createNonce();
		$ip = $_SERVER['REMOTE_ADDR'];
		$name = $this->conn->real_escape_string($user_name);
		$sql = "INSERT INTO users (name, ip, nonce) VALUES ('{$name}', '{$ip}', '{$nonce}');";
		$this->execQuery($sql);
		$insert_id = $this->conn->insert_id;
		if ($this->conn->connect_error) {
		    //die("Connection failed: " . $conn->connect_error);
		    $this->errors->add($conn->connect_error);
		} 

		$user = array(
			'id'=>$insert_id,
			'name'=>$name,
			'nonce'=>$nonce
		);

		return $user;
	}


	private function execQuery($sql) {
		$result = $this->conn->query($sql);
		$this->conn->store_result();
		$rows = NULL;
		if (isset($result->num_rows)) {
			$rows = array();
			$i = 0;
		    while($row = $result->fetch_assoc()) {		    	
		        $rows[$i] = $row;
		        $i++;
		    }
		} 		

		return $rows;
	}

	public static function createNonce() {
	    $id = 1; // replace with actual username
	    $nonce = hash('md4', self::randomString(10)); //not 'secure' but fast

	    return $nonce;
	}


	private static function randomString($length = 10) {
	    $characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
	    $charactersLength = strlen($characters);
	    $randomString = '';
	    for ($i = 0; $i < $length; $i++) {
	        $randomString .= $characters[rand(0, $charactersLength - 1)];
	    }
	    return $randomString;
	}
}



?>