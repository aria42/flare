#!/usr/bin/env perl
#

use strict;
use warnings;

my %words;

foreach my $filename (@ARGV) {
  open(my $fh, '<:encoding(UTF-8)', $filename)
    or die "Could not open file '$filename' $!";

  while (my $line = <$fh>) {
    chomp $line;
    my @words_in_line = split(/\s+/, $line);
    foreach my $word (@words_in_line) {
      $words{$word} = 1;
    }
  }
}

while (my $embedding = <STDIN>) {
  my @parts = split(/ /, $embedding);
  my $word = $parts[0];
  print $embedding if exists $words{$word};
}
