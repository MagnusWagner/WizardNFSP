<template>
  <div class="main">
    <h1 id = "header">Wizard Game</h1>
    <span v-if="starting_hand.length>0">{{starting_hand}}</span>
    <br>
    <span>{{test2}}</span>
    <br>
    <button v-on:click="this.getStartingHand()">Get Starting Hand</button>
    <button v-on:click="this.postCardDecision()">Send Card Index</button>
  </div>
</template>

<script>
import {hostname} from '../constants'


export default {
  name: 'Main',
  data() { return {
    starting_hand: ["g-1","b-2"],
    current_hand: [],
    opponent_played: "",
    current_score: [],
    game_over: false,
    selected_card_index: 5,
    test2:"Nothing yet"
  }
  },
  methods: {
    // Get the possible districts that can be selected.
    getStartingHand: function (){
      fetch(hostname + "/api/current_hand") /// fetch docs: https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
        .then(response => response.json())
        .then(response => {
            console.log(response.toString())
            this.starting_hand = response.starting_hand
        })
    },
    postCardDecision: function() {
      fetch(hostname + '/api/selected_card_index', {
          method: 'POST', // or 'PUT'
          headers: {
            'content-type': 'application/json',
          },
          body: JSON.stringify({"selected_card_index":this.selected_card_index}),
        })
        .then(response => response.json())
        .then(response => {
          this.test2 = response.tester
            // this.opponent_played = response.opponent_played
            // this.current_score = response.current_score
            // this.game_over = response.game_over
    }) 
  }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>

</style>
