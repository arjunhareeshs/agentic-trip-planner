import { useEffect, useRef, useState } from 'react';
import { motion, useScroll, useTransform, useInView } from 'motion/react';
import { useNavigate } from 'react-router';
import airplaneWindow from '../../assets/e85ac945be70074188b70973ecb1941b7cdeef0f.png';

// Section component with scroll animations
function AnimatedSection({
  children,
  className = "",
  style
}: {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  // Ensure position is always set for scroll tracking (merge style first, then ensure position)
  const mergedStyle: React.CSSProperties = {
    ...style,
    position: style?.position || 'relative' as const
  };

  return (
    <motion.section
      ref={ref}
      initial={{ opacity: 0, y: 50 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className={className}
      style={mergedStyle}
    >
      {children}
    </motion.section>
  );
}

export default function Home() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollProgress, setScrollProgress] = useState(0);
  const navigate = useNavigate();
  const citiesRef = useRef<HTMLDivElement>(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end start"]
  });

  useEffect(() => {
    const unsubscribe = scrollYProgress.on('change', (latest) => {
      setScrollProgress(latest);
    });
    return () => unsubscribe();
  }, [scrollYProgress]);

  // Transform values for the airplane window zoom effect
  const windowScale = useTransform(scrollYProgress, [0, 0.4], [1, 2.8]);
  const windowOpacity = useTransform(scrollYProgress, [0, 0.2], [1, 0]);
  const interiorOpacity = useTransform(scrollYProgress, [0, 0.15], [1, 0]);
  const skyOpacity = useTransform(scrollYProgress, [0.15, 0.35], [0, 1]);

  // Transform values for initial content (fades out)
  const initialContentOpacity = useTransform(scrollYProgress, [0, 0.2], [1, 0]);

  // Transform values for second section
  const contentY = useTransform(scrollYProgress, [0.3, 0.5], [80, 0]);
  const contentOpacity = useTransform(scrollYProgress, [0.3, 0.5], [0, 1]);

  const handleBeginJourney = () => {
    navigate('/vibe');
  };

  return (
    <div className="w-full flex flex-col items-center bg-neutral-900">
      {/* HERO SECTION */}
      <section
        ref={containerRef}
        className="w-full h-[200vh]"
        style={{ position: 'relative' }}
      >
        <div className="sticky top-0 w-full h-screen overflow-hidden flex items-center justify-center bg-neutral-900">

          {/* Photorealistic Airplane Window Background Image */}
          <motion.div
            style={{ opacity: interiorOpacity, scale: windowScale }}
            className="absolute inset-0 z-20"
          >
            <img
              src={airplaneWindow}
              alt="Airplane window view"
              className="w-full h-full object-cover"
            />
          </motion.div>

          {/* Initial Hero Content - Visible from the start on top of airplane photo */}
          <motion.div
            style={{ opacity: initialContentOpacity, pointerEvents: scrollProgress > 0.2 ? 'none' : 'auto' }}
            className="absolute inset-0 z-30 flex items-center"
          >
            <div className="container max-w-7xl mx-auto px-12">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">

                {/* Left Side - Main Headline */}
                <div className="space-y-6">
                  <h1 className="text-7xl font-bold text-white leading-tight">
                    EXPLORE<br />
                    THE WORLD
                  </h1>

                  <p className="text-xl text-white leading-relaxed">
                    Adventure starts here.<br />
                    Where will you go next?
                  </p>

                  <div className="pt-4">
                    <button
                      onClick={handleBeginJourney}
                      className="px-10 py-4 bg-white text-slate-900 text-base font-semibold rounded-full hover:bg-slate-100 transition-all duration-300 transform hover:scale-105 shadow-lg"
                    >
                      BEGIN YOUR JOURNEY
                    </button>
                  </div>
                </div>

                {/* Right Side - Supporting Content */}
                <div className="space-y-6 text-white">
                  <h2 className="text-3xl font-bold">
                    Discover Your Next Adventure
                  </h2>

                  <div className="space-y-4 text-base leading-relaxed">
                    <p>
                      From pristine beaches to majestic mountains, from bustling cities to serene countryside,
                      the world is full of incredible destinations waiting to be explored.
                    </p>

                    <p>
                      Whether you're seeking thrilling experiences, cultural immersion, or peaceful relaxation,
                      your perfect journey begins with a single step. Let us help you plan an unforgettable trip
                      tailored to your dreams.
                    </p>
                  </div>
                </div>

              </div>
            </div>
          </motion.div>

          {/* Sky Background that appears as we zoom through the window */}
          <motion.div
            style={{ opacity: skyOpacity }}
            className="absolute inset-0 bg-gradient-to-b from-sky-400 via-orange-300 to-rose-200 z-10"
          >
            <div className="absolute inset-0">
              {/* Animated clouds */}
              <div className="absolute top-[15%] left-[20%] w-72 h-36 bg-white/50 rounded-full blur-3xl animate-float" />
              <div className="absolute top-[30%] right-[15%] w-96 h-48 bg-white/40 rounded-full blur-3xl animate-float-delayed" />
              <div className="absolute bottom-[25%] left-[30%] w-80 h-40 bg-white/30 rounded-full blur-3xl animate-float-slow" />
              <div className="absolute top-[50%] left-[10%] w-64 h-32 bg-white/35 rounded-full blur-3xl animate-float" />
            </div>
          </motion.div>

          {/* Second Hero Content - Appears during scroll transition */}
          <motion.div
            style={{
              y: contentY,
              opacity: contentOpacity,
              pointerEvents: scrollProgress < 0.3 ? 'none' : 'auto'
            }}
            className="absolute inset-0 z-30"
          >
            <div className="container max-w-7xl mx-auto px-12 h-full flex items-center">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center w-full">

                {/* Left Side - Large Headline */}
                <div className="space-y-8">
                  <div className="space-y-6">
                    <h1 className="text-7xl font-bold text-white drop-shadow-2xl leading-tight">
                      Your Journey<br />
                      Begins Here
                    </h1>
                    <div className="w-24 h-1.5 bg-amber-500 shadow-lg" />
                  </div>

                  <p className="text-xl text-white drop-shadow-lg leading-relaxed max-w-lg">
                    Embark on unforgettable adventures across the globe.
                    Discover new horizons, create lasting memories, and experience the world like never before.
                  </p>

                  <div className="pt-6">
                    <button
                      onClick={handleBeginJourney}
                      className="px-10 py-5 bg-amber-600 text-white text-lg font-semibold rounded-xl hover:bg-amber-700 hover:shadow-2xl transition-all duration-300 transform hover:scale-105 shadow-xl"
                    >
                      BEGIN YOUR JOURNEY
                    </button>
                  </div>
                </div>

                {/* Right Side - Supporting Text */}
                <div className="space-y-6 text-white">
                  <div className="space-y-4 text-base leading-relaxed bg-black/20 backdrop-blur-sm p-8 rounded-2xl">
                    <p className="text-lg font-medium drop-shadow-lg">
                      Where will your wanderlust take you?
                    </p>

                    <p className="drop-shadow-md">
                      From the sun-kissed beaches of tropical paradises to the snow-capped peaks
                      of majestic mountains, every destination tells a unique story.
                    </p>

                    <p className="drop-shadow-md">
                      Whether you're seeking thrilling adventures, peaceful retreats, cultural
                      immersion, or culinary delights—the perfect journey awaits.
                    </p>

                    <p className="text-amber-200 italic drop-shadow-md">
                      "The world is a book, and those who do not travel read only one page."
                    </p>
                  </div>
                </div>

              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* CITIES SECTION */}
      <AnimatedSection
        className="relative w-full min-h-screen bg-gradient-to-br from-slate-800 via-slate-700 to-blue-900"
      >
        <div ref={citiesRef} className="container max-w-7xl mx-auto px-12 py-20">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            {/* Left Side - City Travel Text */}
            <div className="space-y-8">
              <div className="space-y-4">
                <h2 className="text-5xl font-bold text-amber-50">
                  Explore Vibrant Cities
                </h2>
                <div className="w-20 h-1 bg-amber-500" />
              </div>

              <div className="space-y-5 text-base text-slate-200">
                <p>
                  Immerse yourself in the energy of iconic skylines, historic landmarks,
                  and bustling streets where culture comes alive around every corner.
                </p>
                <p>
                  From architectural marvels to vibrant nightlife, from local cuisine to
                  world-class museums, each city offers its own unique story waiting to be discovered.
                </p>
                <p>
                  Experience the pulse of urban life and find your perfect metropolitan escape.
                </p>
              </div>
            </div>

            {/* Right Side - City Image */}
            <div className="relative h-[450px] rounded-xl overflow-hidden shadow-2xl">
              <img
                src="https://images.unsplash.com/photo-1757843298369-6e5503c14bfd?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx1cmJhbiUyMHNreWxpbmUlMjBuaWdodCUyMGNpdHklMjBsaWdodHN8ZW58MXx8fHwxNzcxODMwNzMwfDA&ixlib=rb-4.1.0&q=80&w=1080"
                alt="Urban skyline"
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-slate-900/50 to-transparent" />
            </div>
          </div>
        </div>
      </AnimatedSection>

      {/* SEASONS SECTION */}
      <AnimatedSection
        className="relative w-full min-h-screen bg-gradient-to-br from-teal-50 via-sky-50 to-amber-50"
      >
        <div className="container max-w-7xl mx-auto px-12 py-20">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            {/* Left Side - Season Image */}
            <div className="relative h-[450px] rounded-xl overflow-hidden shadow-2xl lg:order-1">
              <img
                src="https://images.unsplash.com/photo-1602260263509-7ca6d705b9d3?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmb3VyJTIwc2Vhc29ucyUyMG5hdHVyZSUyMGNvbGxhZ2V8ZW58MXx8fHwxNzcxODMwNzMwfDA&ixlib=rb-4.1.0&q=80&w=1080"
                alt="Seasonal landscapes"
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-teal-900/20 to-transparent" />
            </div>

            {/* Right Side - Seasonal Text */}
            <div className="space-y-8 lg:order-2">
              <div className="space-y-4">
                <h2 className="text-5xl font-bold text-slate-900">
                  Journey Through Seasons
                </h2>
                <div className="w-20 h-1 bg-teal-600" />
              </div>

              <div className="space-y-5 text-base text-slate-700">
                <p>
                  Every season paints the world in new colors. From cherry blossoms in spring
                  to golden autumn leaves, each time of year offers unique experiences.
                </p>
                <p>
                  Discover sun-drenched summer beaches, cozy winter wonderlands,
                  vibrant fall foliage, and blooming spring gardens.
                </p>
                <p>
                  Let the rhythm of nature guide your journey and explore destinations
                  at their most beautiful moments throughout the year.
                </p>
              </div>
            </div>
          </div>
        </div>
      </AnimatedSection>

      {/* CATEGORIES SECTION */}
      <AnimatedSection
        className="relative w-full min-h-screen bg-gradient-to-br from-orange-50 via-amber-50 to-yellow-50"
      >
        <div className="container max-w-7xl mx-auto px-12 py-20">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            {/* Left Side - Category Text */}
            <div className="space-y-8">
              <div className="space-y-4">
                <h2 className="text-5xl font-bold text-slate-900">
                  Travel Your Way
                </h2>
                <div className="w-20 h-1 bg-orange-600" />
              </div>

              <div className="space-y-5 text-base text-slate-700">
                <p>
                  Whether you're seeking heart-pounding adventures, peaceful retreats,
                  family-friendly fun, or luxurious indulgence, we have the perfect trip for you.
                </p>
                <p>
                  Browse by adventure level, budget, travel style, or special interests
                  to find experiences tailored to your preferences.
                </p>
                <p>
                  From backpacking expeditions to five-star resorts, romantic getaways
                  to group tours—your ideal journey awaits.
                </p>
              </div>
            </div>

            {/* Right Side - Category Image */}
            <div className="relative h-[450px] rounded-xl overflow-hidden shadow-2xl">
              <img
                src="https://images.unsplash.com/photo-1585675444601-25e0f56bc7c1?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxhZHZlbnR1cmUlMjB0cmF2ZWwlMjBiYWNrcGFjayUyMGhpa2luZ3xlbnwxfHx8fDE3NzE4MzA3MzF8MA&ixlib=rb-4.1.0&q=80&w=1080"
                alt="Adventure travel"
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-orange-900/30 to-transparent" />
            </div>
          </div>
        </div>
      </AnimatedSection>

      {/* FOOTER SECTION */}
      <footer
        className="w-full bg-slate-950 text-slate-200"
      >
        <div className="container max-w-7xl mx-auto px-12 py-16">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">
            {/* Company Info */}
            <div className="space-y-4">
              <h3 className="text-2xl font-bold text-amber-400">Wanderlust Travel</h3>
              <p className="text-slate-400">
                Your gateway to unforgettable adventures around the world.
              </p>
            </div>

            {/* Contact Information */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-amber-300">Contact Us</h4>
              <div className="space-y-2 text-slate-400 text-sm">
                <p>Phone: +1 (555) 123-4567</p>
                <p>Toll Free: 1-800-TRAVEL-NOW</p>
                <p>Email: info@wanderlust.com</p>
                <p>Support: support@wanderlust.com</p>
              </div>
            </div>

            {/* Address */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-amber-300">Visit Us</h4>
              <address className="text-slate-400 not-italic text-sm">
                123 Adventure Boulevard<br />
                Travel District<br />
                San Francisco, CA 94102<br />
                United States
              </address>
            </div>

            {/* Social Links */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-amber-300">Follow Us</h4>
              <div className="space-y-2 text-sm">
                <a href="#" className="block text-slate-400 hover:text-amber-400 transition-colors">
                  Facebook
                </a>
                <a href="#" className="block text-slate-400 hover:text-amber-400 transition-colors">
                  Instagram
                </a>
                <a href="#" className="block text-slate-400 hover:text-amber-400 transition-colors">
                  Twitter
                </a>
                <a href="#" className="block text-slate-400 hover:text-amber-400 transition-colors">
                  LinkedIn
                </a>
              </div>
            </div>
          </div>

          {/* Bottom Bar */}
          <div className="mt-12 pt-8 border-t border-slate-800">
            <div className="flex flex-col md:flex-row justify-between items-center gap-4">
              <p className="text-slate-400 text-sm">
                © 2026 Wanderlust Travel. All rights reserved.
              </p>
              <div className="flex gap-6 text-sm text-slate-400">
                <a href="#" className="hover:text-amber-400 transition-colors">Privacy Policy</a>
                <a href="#" className="hover:text-amber-400 transition-colors">Terms of Service</a>
                <a href="#" className="hover:text-amber-400 transition-colors">Cookie Policy</a>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}