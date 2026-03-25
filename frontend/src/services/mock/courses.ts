import type { Course, VideoClip } from '../../types';

export const courses: Course[] = [
  {
    id: 'vase-making-101',
    title: 'Vase Making Fundamentals',
    description:
      'Learn the essential techniques of vase making, from centering clay to pulling walls and shaping elegant forms.',
    difficulty: 'beginner',
    totalClips: 30,
    estimatedTime: '4h 30m',
    progress: 65,
    thumbnail: '/course-vase.jpg',
    category: 'Ceramics',
    instructor: 'Maria Santos',
  },
  {
    id: 'wood-carving-basics',
    title: 'Wood Carving Essentials',
    description:
      'Master fundamental wood carving techniques including chip carving, relief carving, and finishing methods.',
    difficulty: 'beginner',
    totalClips: 30,
    estimatedTime: '5h 00m',
    progress: 30,
    thumbnail: '/course-wood.jpg',
    category: 'Woodwork',
    instructor: 'James Chen',
  },
  {
    id: 'glass-blowing-intro',
    title: 'Introduction to Glass Blowing',
    description:
      'Discover the art of glass blowing with hands-on techniques for creating beautiful glass pieces.',
    difficulty: 'intermediate',
    totalClips: 30,
    estimatedTime: '6h 00m',
    progress: 15,
    thumbnail: '/course-glass.jpg',
    category: 'Glasswork',
    instructor: 'Elena Volkov',
  },
  {
    id: 'leather-crafting',
    title: 'Leather Crafting Mastery',
    description:
      'Explore advanced leather crafting skills from cutting and stitching to tooling and finishing.',
    difficulty: 'intermediate',
    totalClips: 30,
    estimatedTime: '5h 30m',
    progress: 0,
    thumbnail: '/course-leather.jpg',
    category: 'Leatherwork',
    instructor: 'Thomas Wright',
  },
  {
    id: 'jewelry-making',
    title: 'Fine Jewelry Making',
    description:
      'Create stunning jewelry pieces with professional techniques in metalwork, stone setting, and design.',
    difficulty: 'advanced',
    totalClips: 30,
    estimatedTime: '8h 00m',
    progress: 0,
    thumbnail: '/course-jewelry.jpg',
    category: 'Jewelry',
    instructor: 'Sofia Martinez',
  },
  {
    id: 'pottery-wheel',
    title: 'Pottery Wheel Techniques',
    description:
      'Advanced pottery wheel techniques for creating complex forms, trimming, and decorative methods.',
    difficulty: 'advanced',
    totalClips: 1,
    estimatedTime: '7h 00m',
    progress: 0,
    thumbnail: '/course-pottery.jpg',
    category: 'Ceramics',
    instructor: 'David Kim',
  },
];

const vaseMakingTechniques = [
  'Centering the Clay',
  'Opening the Form',
  'Pulling the Walls',
  'Shaping the Belly',
  'Narrowing the Neck',
  'Forming the Lip',
  'Creating the Base',
  'Wall Thickness Control',
  'Symmetry Techniques',
  'Compression Methods',
  'Rib Tool Usage',
  'Sponge Technique',
  'Wire Cut Release',
  'Trimming the Foot',
  'Adding a Foot Ring',
  'Surface Smoothing',
  'Decorative Ridges',
  'Handle Attachment',
  'Spout Formation',
  'Lid Fitting',
  'Drying Preparation',
  'Bisque Firing Prep',
  'Glaze Application',
  'Dip Glazing',
  'Brush Glazing',
  'Wax Resist Technique',
  'Underglaze Decoration',
  'Kiln Loading',
  'Final Inspection',
  'Gallery Finish',
];

const woodCarvingTechniques = [
  'Selecting Wood Grain',
  'Marking the Pattern',
  'Rough Cut Outline',
  'Stop Cut Basics',
  'Chip Carving Triangles',
  'Relief Carving Depth',
  'Gouge Technique',
  'V-Tool Detailing',
  'Rounding Edges',
  'Undercutting Forms',
  'Background Removal',
  'Surface Leveling',
  'Fine Detail Work',
  'Lettering Basics',
  'Geometric Patterns',
  'Floral Motifs',
  'Animal Forms',
  'Face Carving',
  'Bark Inclusion',
  'Pierced Carving',
  'Power Tool Integration',
  'Sanding Techniques',
  'Oil Finishing',
  'Wax Application',
  'Staining Methods',
  'Sealer Application',
  'Tool Sharpening',
  'Safety Practices',
  'Project Assembly',
  'Final Presentation',
];

const potteryWheelClips: VideoClip[] = [
  {
    id: 'pottery-wheel-clip-1',
    title: 'Pottery Expert Demo',
    duration: 10,
    description:
      'This is the real pottery expert video currently available in the system. Choose it to start side-by-side comparison in Compare Studio.',
    thumbnail: '/course-pottery.jpg',
    expertVideoUrl: '/storage/expert/pottery.mp4',
    keyPoints: [
      'Watch the real expert demo currently stored in the backend',
      'Use this clip as the expert reference on the left side of Compare Studio',
      'Upload your learner video on the right and start comparison',
    ],
  },
];

export function getClipsForCourse(courseId: string): VideoClip[] {
  if (courseId === 'pottery-wheel') {
    return potteryWheelClips;
  }

  let techniques: string[];

  switch (courseId) {
    case 'vase-making-101':
      techniques = vaseMakingTechniques;
      break;
    case 'wood-carving-basics':
      techniques = woodCarvingTechniques;
      break;
    default:
      techniques = Array.from(
        { length: 30 },
        (_, i) => `Technique ${i + 1}`,
      );
  }

  return techniques.map((name, i) => ({
    id: `${courseId}-clip-${i + 1}`,
    title: name,
    duration: 120 + Math.floor(Math.random() * 180),
    description: `Learn the ${name.toLowerCase()} technique with step-by-step guidance and expert tips.`,
    thumbnail: `/clip-${courseId}-${i + 1}.jpg`,
    keyPoints: [
      `Understand the basics of ${name.toLowerCase()}`,
      'Watch the instructor demonstrate proper form',
      'Practice the technique at your own pace',
    ],
  }));
}
