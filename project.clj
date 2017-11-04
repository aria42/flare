(defproject aria42/flare "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :global-vars {*warn-on-reflection* true}
  :jvm-opts ["-mx2000m"]
  :aot :all
  :lein-release {:deploy-via :clojars}
  :scm {:name "git"
        :url "https://github.com/aria42/flare"}
  :profiles {:dev {:global-vars {*warn-on-reflection* true
                                 *unchecked-math* :warn-on-boxed}
                   :dependencies [[criterium "0.4.4"]]}}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [uncomplicate/neanderthal "0.17.0"]
                 [prismatic/schema "1.1.6"]
                 [prismatic/plumbing "0.5.4"]
                 [org.clojure/tools.cli "0.3.5"]])
