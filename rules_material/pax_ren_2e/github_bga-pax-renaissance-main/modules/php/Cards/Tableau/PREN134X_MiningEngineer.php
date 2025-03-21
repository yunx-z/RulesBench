<?php

namespace PaxRenaissance\Cards\Tableau;

class PREN134X_MiningEngineer extends \PaxRenaissance\Models\TableauCard
{
  public function __construct($row)
  {
    parent::__construct($row);
    $this->id = 'PREN134X_MiningEngineer';
    $this->empire = WEST;
    $this->expansionCard = true;
    $this->flavorText = [
      clienttranslate("Georgius Agricola, scholar, chemist, and mining engineer."),
      clienttranslate("His lavishly illustrated masterpiece, De re metallica, recorded Saxon mining technology, the best in the world."),
      clienttranslate("A staunch Catholic, he suffered persecution by the Lutherans of Chemnitz."),
    ];
    $this->name = clienttranslate('Mining Engineer');
    $this->prestige = [CATHOLIC];
    $this->region = WEST;
    $this->specialAbilities = [
      [
        'id' => SA_FREE_WESTERN_OPS,
        'title' => clienttranslate('FORTIFICATIONS'),
        'text' => [
          'log' => clienttranslate('You may do "activate western ops" without expending an action.'),
          'args' => [],
        ],
        'abilityAction' => true,
        'top' => 159,
        'left' => 108,
        'height' => 52,
        'width' => 36,
      ]
    ];
  }
}
